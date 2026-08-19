import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchHealth,
  fetchJobStatus,
  startImagesJob,
  startVideoJob,
} from './api'
import Header from './components/Header'
import ImageResultPanel from './components/ImageResultPanel'
import ModeSelector from './components/ModeSelector'
import ResultPanel from './components/ResultPanel'
import UploadZone from './components/UploadZone'
import type {
  HealthResponse,
  MediaType,
  Mode,
  ProcessResult,
} from './types'
import { API_BASE, isVideoResult } from './types'

type Status = 'empty' | 'loading' | 'error' | 'done'

// 用 sessionStorage 记录进行中的任务，刷新后检测残留并自动终止
const ACTIVE_JOB_KEY = 'active_job_id'

export default function App() {
  const [dark, setDark] = useState(() =>
    localStorage.getItem('theme') === 'dark',
  )
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [mediaType, setMediaType] = useState<MediaType>('video')
  const [mode, setMode] = useState<Mode>('segmentation')
  const [status, setStatus] = useState<Status>('empty')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ProcessResult | null>(null)
  const [progress, setProgress] = useState(0)
  const [frameProgress, setFrameProgress] = useState<{
    processed: number
    total: number
  } | null>(null)

  const pollRef = useRef<number | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [showCancelModal, setShowCancelModal] = useState(false)

  // 刷新/重新打开页面后：若存在未完成的处理任务，自动终止（系统弹窗在
  // 部分嵌入式浏览器中会被禁止，这里做兜底，确保任务一定被取消）
  useEffect(() => {
    const leftover = sessionStorage.getItem(ACTIVE_JOB_KEY)

    if (!leftover) return

    sessionStorage.removeItem(ACTIVE_JOB_KEY)

    fetch(`${API_BASE}/api/process-cancel/${leftover}`, {
      method: 'POST',
    }).catch(() => {})

    setError('检测到上次刷新时仍在处理的视频任务，已自动终止。')
    setStatus('error')
  }, [])

  useEffect(() => {
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current)
    }
  }, [])

  // 处理过程中刷新/关闭页面：先弹窗确认；确认离开则通知后端终止任务
  useEffect(() => {
    if (status !== 'loading' || !jobId) return

    const cancelUrl = `${API_BASE}/api/process-cancel/${jobId}`

    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ''
    }

    const onPageHide = () => {
      navigator.sendBeacon(cancelUrl)
    }

    window.addEventListener('beforeunload', onBeforeUnload)
    window.addEventListener('pagehide', onPageHide)

    return () => {
      window.removeEventListener('beforeunload', onBeforeUnload)
      window.removeEventListener('pagehide', onPageHide)
    }
  }, [status, jobId])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  const handleRun = useCallback(async () => {
    if (files.length === 0) {
      setError(mediaType === 'image' ? '请先上传图片' : '请先上传视频')
      setStatus('error')
      return
    }

    setStatus('loading')
    setError(null)
    setResult(null)
    setProgress(0)
    setFrameProgress(null)
    setJobId(null)

    if (pollRef.current !== null) window.clearInterval(pollRef.current)

    // 图片批量 / 视频统一走任务式轮询（图片进度按已处理张数推进）
    try {
      const { job_id } =
        mediaType === 'image'
          ? await startImagesJob(mode, files)
          : await startVideoJob(mode, files[0])

      setJobId(job_id)
      sessionStorage.setItem(ACTIVE_JOB_KEY, job_id)

      pollRef.current = window.setInterval(async () => {
        try {
          const s = await fetchJobStatus(job_id)

          setProgress(Math.round((s.progress ?? 0) * 100))

          if (s.processed != null && s.total != null) {
            setFrameProgress({
              processed: s.processed,
              total: s.total,
            })
          }

          if (s.status === 'done' && s.result) {
            if (pollRef.current !== null) {
              window.clearInterval(pollRef.current)
              pollRef.current = null
            }
            sessionStorage.removeItem(ACTIVE_JOB_KEY)
            setJobId(null)
            setResult(s.result)
            setProgress(100)
            setStatus('done')
          } else if (s.status === 'error') {
            if (pollRef.current !== null) {
              window.clearInterval(pollRef.current)
              pollRef.current = null
            }
            sessionStorage.removeItem(ACTIVE_JOB_KEY)
            setJobId(null)
            setError(s.error || '视频处理失败')
            setStatus('error')
          } else if (s.status === 'cancelled') {
            if (pollRef.current !== null) {
              window.clearInterval(pollRef.current)
              pollRef.current = null
            }
            sessionStorage.removeItem(ACTIVE_JOB_KEY)
            setJobId(null)
            setError('处理已被取消')
            setStatus('error')
          }
        } catch {
          // 网络抖动时继续轮询
        }
      }, 800)
    } catch (err) {
      setError(err instanceof Error ? err.message : '视频处理失败')
      setStatus('error')
    }
  }, [files, mediaType, mode])

  // 确认停止：调用后端取消接口终止任务，再清理轮询与本地记录
  const confirmStop = useCallback(async () => {
    setShowCancelModal(false)

    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }

    if (jobId) {
      try {
        await fetch(`${API_BASE}/api/process-cancel/${jobId}`, {
          method: 'POST',
        })
      } catch {
        // 后端不可达时任务可能仍在跑，下次刷新由兜底逻辑终止
      }
    }

    sessionStorage.removeItem(ACTIVE_JOB_KEY)
    setJobId(null)
    setError('处理已停止')
    setStatus('error')
  }, [jobId])

  // health 未返回时视为未知，避免“不可用”徽标闪动
  const yoloAvailable =
    health === null ? null : health.yolo_available

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      <Header
        health={health}
        dark={dark}
        onToggleDark={() => setDark((d) => !d)}
      />

      <main className="mx-auto max-w-5xl px-4 py-8">
        {health && health.status !== 'ok' && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
            后端模型未就绪：{health.message ?? '未知错误'}
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-[380px_1fr]">
          {/* 控制区 */}
          <section className="space-y-6">
            <div>
              <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Video Segmentation Demo
              </h1>
              <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                上传道路图片或视频，使用 SegFormer 进行实时道路分割。
              </p>
            </div>

            <div className="space-y-2">
              <p className="text-xs font-medium uppercase tracking-wide text-zinc-400 dark:text-zinc-500">
                1 · Media Upload
              </p>
              <div className="grid grid-cols-2 gap-2 rounded-lg border border-zinc-200 bg-zinc-100 p-1 dark:border-zinc-800 dark:bg-zinc-900">
                {(['video', 'image'] as MediaType[]).map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setMediaType(t)}
                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                      mediaType === t
                        ? 'bg-white text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-zinc-100'
                        : 'text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200'
                    }`}
                  >
                    {t === 'video' ? '视频' : '图片'}
                  </button>
                ))}
              </div>
              <UploadZone
                files={files}
                mediaType={mediaType}
                onFiles={setFiles}
              />
            </div>

            <div className="space-y-2">
              <p className="text-xs font-medium uppercase tracking-wide text-zinc-400 dark:text-zinc-500">
                2 · Mode
              </p>
              <ModeSelector
                mode={mode}
                yoloAvailable={yoloAvailable}
                onChange={setMode}
              />
              {mode === 'fusion' && yoloAvailable === false && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  YOLO 权重缺失：融合模式将退回 SegFormer 分割结果。
                </p>
              )}
            </div>

            <button
              type="button"
              onClick={handleRun}
              disabled={status === 'loading'}
              className="w-full rounded-xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {status === 'loading' ? 'Processing…' : 'Start Analysis'}
            </button>

            {status === 'loading' && (
              <div className="space-y-2 rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-3 dark:border-zinc-800 dark:bg-zinc-900">
                <div className="flex items-center gap-2">
                  <svg
                    className="h-4 w-4 animate-spin text-emerald-600"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4Z"
                    />
                  </svg>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {mediaType === 'image'
                      ? '正在批量处理图片…'
                      : '正在逐帧推理并生成结果视频…'}
                  </p>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
                  <div
                    className="h-full rounded-full bg-emerald-600 transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-right font-mono text-[11px] text-zinc-400 dark:text-zinc-500">
                  {frameProgress
                    ? `${frameProgress.processed} / ${frameProgress.total} ${mediaType === 'image' ? '张' : '帧'} · `
                    : ''}
                  {progress}%
                </p>
                <button
                  type="button"
                  onClick={() => setShowCancelModal(true)}
                  className="w-full rounded-lg border border-red-300 px-3 py-1.5 text-xs font-medium text-red-600 transition-colors hover:bg-red-50 dark:border-red-500/40 dark:text-red-400 dark:hover:bg-red-500/10"
                >
                  停止处理
                </button>
              </div>
            )}

            {status === 'error' && (
              <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
                {error}
              </p>
            )}
          </section>

          {/* 结果区 */}
          <section className="min-w-0">
            {status === 'empty' && (
              <div className="grid h-full min-h-[320px] place-items-center rounded-xl border border-dashed border-zinc-300 dark:border-zinc-700">
                <div className="text-center">
                  <svg
                    className="mx-auto h-10 w-10 text-zinc-300 dark:text-zinc-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
                    />
                  </svg>
                  <p className="mt-3 text-sm text-zinc-400 dark:text-zinc-500">
                    上传图片或视频并点击 Start Analysis 查看分割结果
                  </p>
                </div>
              </div>
            )}

            {status === 'loading' && (
              <div className="grid h-full min-h-[320px] place-items-center rounded-xl border border-dashed border-zinc-300 dark:border-zinc-700">
                <div className="w-full max-w-sm px-6 text-center">
                  <svg
                    className="mx-auto h-8 w-8 animate-spin text-emerald-600"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4Z"
                    />
                  </svg>
                  <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400">
                    Processing… {progress}%
                  </p>
                  <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
                    <div
                      className="h-full rounded-full bg-emerald-600 transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              </div>
            )}

            {status === 'done' && result &&
              (isVideoResult(result) ? (
                <ResultPanel result={result} />
              ) : (
                <ImageResultPanel result={result} />
              ))}
          </section>
        </div>
      </main>

      <footer className="border-t border-zinc-200 py-6 dark:border-zinc-800">
        <p className="text-center text-xs text-zinc-400 dark:text-zinc-500">
          SegFormer Video Segmentation · FastAPI + React Demo · 真实模型推理
        </p>
      </footer>

      {/* 处理中退出确认弹窗（应用内，不受浏览器系统弹窗限制） */}
      {showCancelModal && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/50 px-4">
          <div className="w-full max-w-sm rounded-2xl border border-zinc-200 bg-white p-6 shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
            <h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
              处理仍在进行
            </h3>
            <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
              视频处理已完成 {progress}%，正在逐帧推理。
              退出或刷新页面将终止当前任务，确定要终止吗？
            </p>
            <div className="mt-5 flex gap-3">
              <button
                type="button"
                onClick={() => setShowCancelModal(false)}
                className="flex-1 rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50 dark:border-zinc-600 dark:text-zinc-200 dark:hover:bg-zinc-800"
              >
                继续处理
              </button>
              <button
                type="button"
                onClick={confirmStop}
                className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-500"
              >
                终止并退出
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
