import { useCallback, useEffect, useRef, useState } from 'react'
import type { MediaType } from '../types'

interface UploadZoneProps {
  files: File[]
  mediaType: MediaType
  onFiles: (files: File[]) => void
}

const ACCEPT: Record<MediaType, string> = {
  video: 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska',
  image: 'image/png,image/jpeg,image/webp,image/bmp',
}

export default function UploadZone({
  files,
  mediaType,
  onFiles,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)

  // 组件卸载或切换新文件后再释放旧 blob URL，
  // 避免仍在加载/播放的预览被中断（ERR_ABORTED）
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  // 切换媒体类型时清空已选文件
  useEffect(() => {
    onFiles([])
    setPreview(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mediaType])

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list || list.length === 0) return

      if (mediaType === 'video') {
        const f = list[0]
        const ok =
          f.type.startsWith('video/') || f.name.toLowerCase().endsWith('.mp4')
        if (!ok) return
        onFiles([f])
        setPreview(URL.createObjectURL(f))
        return
      }

      // 图片：批量多选
      const imgs = Array.from(list).filter(
        (f) => f.type.startsWith('image/'),
      )
      if (imgs.length === 0) return
      onFiles(imgs)
      setPreview(null)
    },
    [mediaType, onFiles],
  )

  const clear = useCallback(() => {
    onFiles([])
    setPreview(null)
  }, [onFiles])

  const hint =
    mediaType === 'video'
      ? 'MP4 · 支持完整视频时长'
      : 'JPG / PNG / WEBP / BMP · 可多选批量处理'

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT[mediaType]}
        multiple={mediaType === 'image'}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {files.length === 0 ? (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            handleFiles(e.dataTransfer.files)
          }}
          className={`flex w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors ${
            dragging
              ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-500/10'
              : 'border-zinc-300 bg-zinc-50 hover:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-900/50 dark:hover:border-zinc-600'
          }`}
        >
          <svg
            className="h-8 w-8 text-zinc-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
            />
          </svg>
          <div>
            <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              {mediaType === 'video'
                ? '点击或拖拽上传视频'
                : '点击或拖拽上传图片（可多选）'}
            </p>
            <p className="mt-1 text-xs text-zinc-400">{hint}</p>
          </div>
        </button>
      ) : (
        <div className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
          {preview && (
            <video
              src={preview}
              controls
              preload="metadata"
              className="max-h-72 w-full bg-zinc-100 dark:bg-zinc-900"
            />
          )}
          {mediaType === 'image' && files.length > 0 && (
            <div className="max-h-40 overflow-y-auto border-b border-zinc-200 bg-zinc-50 px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
              <p className="mb-2 text-xs font-medium text-zinc-500 dark:text-zinc-400">
                已选择 {files.length} 张图片
              </p>
              <ul className="space-y-1">
                {files.map((f, i) => (
                  <li
                    key={`${f.name}-${i}`}
                    className="truncate font-mono text-[11px] text-zinc-600 dark:text-zinc-400"
                  >
                    {i + 1}. {f.name}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="flex items-center justify-between gap-3 border-t border-zinc-200 bg-zinc-50 px-4 py-2 dark:border-zinc-800 dark:bg-zinc-900">
            <span className="truncate font-mono text-xs text-zinc-600 dark:text-zinc-400">
              {mediaType === 'video'
                ? files[0]?.name
                : `${files.length} 张图片`}
            </span>
            <button
              type="button"
              onClick={clear}
              className="shrink-0 rounded-md border border-zinc-300 px-2 py-0.5 text-xs text-zinc-600 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
            >
              更换
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
