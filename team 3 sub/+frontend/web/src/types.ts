export type Mode = 'segmentation' | 'fusion'
export type MediaType = 'video' | 'image'

export interface HealthResponse {
  status: 'ok' | 'error'
  model_source: string | null
  device: string | null
  yolo_available: boolean
  message: string | null
}

export interface VideoMetrics {
  resolution: string
  fps: number
  frame_count: number
  processing_time_s: number
  avg_inference_time_ms: number
  device: string
  encoder?: string
  decoder?: string
  frames_with_vehicles?: number
}

export interface YoloStatus {
  available: boolean
  message: string
}

export interface VideoProcessResponse {
  mode: Mode
  files: {
    input: string
    output: string
  }
  metrics: VideoMetrics
  model_source: string
  yolo: YoloStatus
}

export interface ImageMetrics {
  resolution: string
  device: string
  vehicle_count: number
}

export interface ImageProcessResponse {
  mode: Mode
  files: {
    input: string
    output: string
  }
  metrics: ImageMetrics
  model_source: string
  yolo: YoloStatus
}

export interface ImageBatchResponse {
  mode: Mode
  results: ImageProcessResponse[]
}

export type ProcessResult = VideoProcessResponse | ImageBatchResponse

export function isVideoResult(
  r: ProcessResult,
): r is VideoProcessResponse {
  return 'results' in r === false
}

export const RESULT_BASE = '/api/results/'

// 后端地址：媒体文件（视频）直连后端，避免开发代理对流媒体/Range 请求的干扰。
// 可通过环境变量 VITE_API_BASE 覆盖（例如部署到其他主机时）。
export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  'http://127.0.0.1:8000'

export function resultUrl(filename: string): string {
  return `${API_BASE}${RESULT_BASE}${filename}`
}
