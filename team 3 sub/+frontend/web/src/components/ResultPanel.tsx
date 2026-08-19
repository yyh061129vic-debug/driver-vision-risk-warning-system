import type { VideoProcessResponse } from '../types'
import { resultUrl } from '../types'

interface ResultPanelProps {
  result: VideoProcessResponse
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

function VideoCard({
  title,
  src,
  badge,
  badgeColor,
}: {
  title: string
  src: string
  badge: string
  badgeColor?: string
}) {
  return (
    <div className="group overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm transition-shadow hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
          {title}
        </h3>
        <span
          className={`rounded-full px-2.5 py-0.5 text-[10px] font-medium ${
            badgeColor ??
            'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400'
          }`}
        >
          {badge}
        </span>
      </div>
      <div className="bg-zinc-100 p-3 dark:bg-zinc-950/50">
        <video
          src={src}
          controls
          preload="metadata"
          className="aspect-video w-full rounded-xl border border-zinc-200 bg-black object-contain dark:border-zinc-800"
        />
      </div>
    </div>
  )
}

export default function ResultPanel({ result }: ResultPanelProps) {
  const { files, metrics } = result

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
          Results
        </h2>
        <span className="rounded-full bg-zinc-100 px-2.5 py-0.5 text-xs text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
          {result.mode === 'fusion' ? 'SegFormer + YOLO' : 'SegFormer'}
        </span>
      </div>

      {/* 视频上下排列，大尺寸自适应 */}
      <div className="space-y-6">
        <VideoCard
          title="Original Video"
          src={resultUrl(files.input)}
          badge="Original"
          badgeColor="bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
        />
        <VideoCard
          title="Processed Video"
          src={resultUrl(files.output)}
          badge={result.mode === 'fusion' ? 'Fusion' : 'SegFormer'}
        />
      </div>

      <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="mb-3 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
          Metrics
        </h3>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          <Metric label="Resolution" value={metrics.resolution} />
          <Metric label="FPS" value={String(metrics.fps)} />
          <Metric label="Frame Count" value={String(metrics.frame_count)} />
          <Metric
            label="Processing Time"
            value={`${metrics.processing_time_s} s`}
          />
          <Metric
            label="Avg Inference"
            value={`${metrics.avg_inference_time_ms} ms`}
          />
          <Metric label="Device" value={metrics.device} />
          {metrics.encoder && (
            <Metric
              label="Encoder"
              value={metrics.encoder === 'nvenc' ? 'NVENC (GPU)' : 'OpenCV'}
            />
          )}
          {metrics.decoder && (
            <Metric
              label="Decoder"
              value={metrics.decoder === 'nvdec' ? 'NVDEC (GPU)' : 'OpenCV'}
            />
          )}
          {metrics.frames_with_vehicles != null && (
            <Metric
              label="Frames w/ Vehicles"
              value={String(metrics.frames_with_vehicles)}
            />
          )}
        </div>
      </div>

      {result.mode === 'fusion' && result.yolo && !result.yolo.available && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-400">
          {result.yolo.message}。Fusion 模式退回 SegFormer 道路分割结果。
        </p>
      )}

      <p className="text-[11px] text-zinc-400 dark:text-zinc-500">
        Model source: {result.model_source}
      </p>
    </div>
  )
}
