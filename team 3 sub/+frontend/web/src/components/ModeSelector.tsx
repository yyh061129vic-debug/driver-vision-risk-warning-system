import type { Mode } from '../types'

interface ModeSelectorProps {
  mode: Mode
  yoloAvailable: boolean | null
  onChange: (mode: Mode) => void
}

const OPTIONS: { value: Mode; label: string; hint: string }[] = [
  { value: 'segmentation', label: 'SegFormer', hint: '道路分割' },
  { value: 'fusion', label: 'SegFormer + YOLO', hint: '融合结果' },
]

export default function ModeSelector({
  mode,
  yoloAvailable,
  onChange,
}: ModeSelectorProps) {
  return (
    <div className="grid grid-cols-2 gap-2 rounded-xl border border-zinc-200 bg-zinc-50 p-1 dark:border-zinc-800 dark:bg-zinc-900">
      {OPTIONS.map((opt) => {
        const active = mode === opt.value
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={`rounded-lg px-3 py-2.5 text-left transition-colors ${
              active
                ? 'bg-white shadow-sm ring-1 ring-zinc-200 dark:bg-zinc-800 dark:ring-zinc-700'
                : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200'
            }`}
          >
            <span className="flex items-center justify-between gap-2">
              <span
                className={`text-sm font-medium ${
                  active
                    ? 'text-zinc-900 dark:text-zinc-100'
                    : 'text-zinc-600 dark:text-zinc-400'
                }`}
              >
                {opt.label}
              </span>
              {opt.value === 'fusion' && yoloAvailable === false && (
                <span className="rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-500/15 dark:text-amber-400">
                  不可用
                </span>
              )}
            </span>
            <span className="mt-0.5 block text-xs text-zinc-400 dark:text-zinc-500">
              {opt.hint}
            </span>
          </button>
        )
      })}
    </div>
  )
}
