import type { HealthResponse } from '../types'

interface HeaderProps {
  health: HealthResponse | null
  dark: boolean
  onToggleDark: () => void
}

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${
        ok ? 'bg-emerald-500' : 'bg-red-500'
      }`}
    />
  )
}

export default function Header({ health, dark, onToggleDark }: HeaderProps) {
  const online = health?.status === 'ok'

  return (
    <header className="sticky top-0 z-20 border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-emerald-600 font-mono text-sm font-semibold text-white">
            S
          </span>
          <div className="leading-tight">
            <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              SegFormer Road Segmentation
            </p>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Video Demo · SegFormer + YOLO Fusion
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {health && (
            <div className="hidden items-center gap-2 rounded-full border border-zinc-200 px-3 py-1 text-xs text-zinc-600 sm:flex dark:border-zinc-800 dark:text-zinc-400">
              <StatusDot ok={online} />
              <span>{online ? 'Backend ready' : 'Backend offline'}</span>
              {health.yolo_available && (
                <span className="text-zinc-400 dark:text-zinc-500">· YOLO</span>
              )}
            </div>
          )}

          <button
            type="button"
            onClick={onToggleDark}
            aria-label="切换深浅色模式"
            className="rounded-lg border border-zinc-200 p-2 text-zinc-600 transition-colors hover:bg-zinc-100 dark:border-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-900"
          >
            {dark ? (
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 3v1.5M12 19.5V21M4.5 12H3m3.2-4.3L5 6.5m14 .2-1.2 1.2M6.2 16.3 5 17.5m14-.2-1.2-1.2M15.75 12a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0Z"
                />
              </svg>
            ) : (
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"
                />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  )
}
