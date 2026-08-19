import type {
  HealthResponse,
  ImageBatchResponse,
  ImageProcessResponse,
  Mode,
  VideoProcessResponse,
} from './types'
import { API_BASE } from './types'

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`)
  return handle<HealthResponse>(res)
}

export interface JobStatus {
  status: 'processing' | 'done' | 'error' | 'cancelled'
  progress: number
  processed?: number
  total?: number
  result?: VideoProcessResponse | ImageBatchResponse
  error?: string | null
}

export async function startVideoJob(
  mode: Mode,
  file: File,
): Promise<{ job_id: string }> {
  const form = new FormData()
  form.append('file', file)
  form.append('mode', mode)

  const res = await fetch(`${API_BASE}/api/process-video`, {
    method: 'POST',
    body: form,
  })

  return handle<{ job_id: string }>(res)
}

export async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/process-status/${jobId}`)
  return handle<JobStatus>(res)
}

export async function startImageJob(
  mode: Mode,
  file: File,
): Promise<ImageProcessResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('mode', mode)

  const res = await fetch(`${API_BASE}/api/process-image`, {
    method: 'POST',
    body: form,
  })

  return handle<ImageProcessResponse>(res)
}

export async function startImagesJob(
  mode: Mode,
  files: File[],
): Promise<{ job_id: string }> {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  form.append('mode', mode)

  const res = await fetch(`${API_BASE}/api/process-images`, {
    method: 'POST',
    body: form,
  })

  return handle<{ job_id: string }>(res)
}
