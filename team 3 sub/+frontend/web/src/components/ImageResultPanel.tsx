import { useState } from 'react'
import type { ImageBatchResponse } from '../types'
import { resultUrl } from '../types'

interface ImageResultPanelProps {
  result: ImageBatchResponse
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white px-3 py-2.5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <p className="text-[11px] text-zinc-400 dark:text-zinc-500">{label}</p>
      <p className="mt-0.5 font-mono text-sm text-zinc-900 dark:text-zinc-100">
        {value}
      </p>
    </div>
  )
}

export default function ImageResultPanel({
  result,
}: ImageResultPanelProps) {
  const { results } = result
  const [selected, setSelected] = useState(0)

  if (results.length === 0) return null

  const item = results[selected]
  const { files, metrics } = item

  const prev = () =>
    setSelected((s) => (s - 1 + results.length) % results.length)
  const next = () => setSelected((s) => (s + 1) % results.length)

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
          Results
        </h2>
        <span className="rounded-full bg-zinc-100 px-2.5 py-0.5 text-xs text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
          {results.length} 张 · {item.mode === 'fusion' ? 'Fusion' : 'SegFormer'}
        </span>
      </div>

      {/* 缩略图画廊：点击快速锁定任意一张 */}
      <div>
        <p className="mb-2 text-xs font-medium text-zinc-400 dark:text-zinc-500">
          点击缩略图查看对应结果
        </p>
        <div className="flex gap-2 overflow-x-auto pb-2">
          {results.map((r, i) => (
            <button
              key={r.files.output}
              type="button"
              onClick={() => setSelected(i)}
              className={`group relative shrink-0 overflow-hidden rounded-lg border-2 transition-all ${
                i === selected
                  ? 'border-emerald-500 shadow-md'
                  : 'border-transparent opacity-75 hover:opacity-100'
              }`}
            >
              <img
                src={resultUrl(r.files.output)}
                alt={`result ${i + 1}`}
                className="h-16 w-24 object-cover"
              />
              <span className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5 text-center text-[10px] font-medium text-white">
                {i + 1}
                {item.mode === 'fusion' && (
                  <span className="ml-1 text-emerald-300">
                    {r.metrics.vehicle_count}辆
                  </span>
                )}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* 主对比区：与视频一致，上下排列大图卡片 */}
      <div className="space-y-6">
        <div className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm transition-shadow hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900">
          <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Original Image
            </h3>
            <span className="rounded-full bg-zinc-100 px-2.5 py-0.5 text-[10px] font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
              {selected + 1} / {results.length}
            </span>
          </div>
          <div className="bg-zinc-100 p-3 dark:bg-zinc-950/50">
            <img
              src={resultUrl(files.input)}
              alt="original"
              className="max-h-[70vh] w-full rounded-xl border border-zinc-200 bg-black object-contain dark:border-zinc-800"
            />
          </div>
        </div>
        <div className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm transition-shadow hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900">
          <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Result Image
            </h3>
            <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400">
              {item.mode === 'fusion' ? 'Fusion' : 'SegFormer'}
            </span>
          </div>
          <div className="bg-zinc-100 p-3 dark:bg-zinc-950/50">
            <img
              src={resultUrl(files.output)}
              alt="result"
              className="max-h-[70vh] w-full rounded-xl border border-zinc-200 bg-black object-contain dark:border-zinc-800"
            />
          </div>
        </div>
      </div>

      {/* 上一张 / 下一张 */}
      <div className="flex items-center justify-center gap-3">
        <button
          type="button"
          onClick={prev}
          className="rounded-lg border border-zinc-300 px-4 py-1.5 text-xs font-medium text-zinc-600 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
        >
          ← 上一张
        </button>
        <span className="font-mono text-xs text-zinc-400">
          {selected + 1} / {results.length}
        </span>
        <button
          type="button"
          onClick={next}
          className="rounded-lg border border-zinc-300 px-4 py-1.5 text-xs font-medium text-zinc-600 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
        >
          下一张 →
        </button>
      </div>

      {/* 指标（当前选中） */}
      <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="mb-3 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
          Metrics
        </h3>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          <Metric label="Resolution" value={metrics.resolution} />
          <Metric label="Device" value={metrics.device} />
          {item.mode === 'fusion' && (
            <Metric
              label="Vehicles Detected"
              value={String(metrics.vehicle_count)}
            />
          )}
        </div>
      </div>

      {item.mode === 'fusion' && item.yolo && !item.yolo.available && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-400">
          {item.yolo.message}。Fusion 模式退回 SegFormer 道路分割结果。
        </p>
      )}

      <p className="text-[11px] text-zinc-400 dark:text-zinc-500">
        Model source: {item.model_source}
      </p>
    </div>
  )
}
