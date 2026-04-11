import type { RunPayload, PipelineResult } from './types'

export async function startRun(payload: RunPayload): Promise<{task_id: string}> {
  const r = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!r.ok) throw new Error('Failed to start run')
  return r.json()
}

export async function fetchResult(taskId: string): Promise<PipelineResult> {
  const r = await fetch(`/api/result/${taskId}`)
  if (!r.ok) throw new Error('Failed to fetch result')
  return r.json()
}
