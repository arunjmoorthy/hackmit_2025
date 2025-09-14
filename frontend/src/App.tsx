import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Snake from './components/Snake'
import AudioRecorder from './components/AudioRecorder'

type CompletePayload = {
  trimmed_video?: string
  history?: string
  remapped_history?: string
  transcript_txt?: string
  transcript_json?: string
  warp?: string
}

type EventItem = { id: string; kind: 'status' | 'error' | 'done' | 'info' | string; message: string }

const backendBase = (import.meta.env.VITE_BACKEND_BASE as string | undefined) || ''

export default function App() {
  const [url, setUrl] = useState('')
  const [description, setDescription] = useState('')
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const USE_VOICE = false // Set to true to use uploaded voice, false for standard voice
  const [submitting, setSubmitting] = useState(false)
  const [events, setEvents] = useState<EventItem[]>([])
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [links, setLinks] = useState<Record<string, string>>({})
  const [progress, setProgress] = useState(0)
  const controllerRef = useRef<AbortController | null>(null)
  const resultRef = useRef<HTMLDivElement | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [showLogs, setShowLogs] = useState(false)
  const [gameStarted, setGameStarted] = useState(false)

  const onRecorded = useCallback((blob: Blob | null) => {
    setAudioBlob(blob)
  }, [])

  const onSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url) return alert('Enter a URL')
    if (!audioBlob) return alert('Record an audio snippet first')
    setSubmitting(true)
    setEvents([])
    setVideoUrl(null)
    setLinks({})
    setProgress(0)
    setShowModal(true)

    const form = new FormData()
    form.append('url', url)
    form.append('description', description)
    form.append('audio', audioBlob, 'input.webm')
    form.append('use_uploaded_voice', String(USE_VOICE))

    controllerRef.current = new AbortController()

    try {
      const resp = await fetch(`${backendBase}/api/process`, {
        method: 'POST',
        body: form,
        headers: { Accept: 'text/event-stream' },
        signal: controllerRef.current.signal,
      })
      if (!resp.ok || !resp.body) throw new Error(`Request failed: ${resp.status}`)

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const pushEvent = (kind: EventItem['kind'], message: string) => {
        setEvents(prev => [...prev, { id: `${Date.now()}-${prev.length}`, kind, message }])
      }

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        let sepIdx: number
        while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
          const raw = buffer.slice(0, sepIdx)
          buffer = buffer.slice(sepIdx + 2)
          const lines = raw.split('\n')
          let eventName = 'message'
          let dataLine = ''
          for (const line of lines) {
            if (line.startsWith('event:')) eventName = line.slice(6).trim()
            if (line.startsWith('data:')) dataLine += line.slice(5).trim()
          }
          try {
            const parsed = JSON.parse(dataLine)
            const payload = parsed.data || parsed
            if (eventName === 'status') {
              pushEvent('status', payload.message || JSON.stringify(payload))
              const m = (payload.message || '') as string
              if (m.includes('Creating the visuals')) setProgress(10)
              if (m.includes('Trimming the video')) setProgress(40)
              if (m.includes('Building transcript')) setProgress(60)
              if (m.includes('Synthesizing voiceover')) setProgress(80)
            } else if (eventName === 'progress') {
              const v = Number((payload.value ?? 0))
              if (!Number.isNaN(v)) setProgress(Math.max(progress, Math.min(100, v)))
            } else if (eventName === 'error') {
              pushEvent('error', payload.message || JSON.stringify(payload))
            } else if (eventName === 'agent_done') {
              pushEvent('done', 'Agent completed')
              setProgress(30)
              setGameStarted(true)
            } else if (eventName === 'trim_done') {
              pushEvent('done', 'Trim completed')
              setProgress(50)
            } else if (eventName === 'transcript_done') {
              pushEvent('done', 'Transcript ready')
              setProgress(70)
            } else if (eventName === 'complete') {
              pushEvent('done', 'All done!')
              const p = payload as CompletePayload
              const final = (payload as any).final_video as string | undefined
              if (final) setVideoUrl(`${backendBase}${final}`)
              else if (p.trimmed_video) setVideoUrl(`${backendBase}${p.trimmed_video}`)
              const l: Record<string, string> = {}
              ;(['history','remapped_history','transcript_txt','transcript_json','warp','captions','mixed_audio','final_video'] as const).forEach(k => {
                const v = (p as any)[k]
                if (v) l[k] = `${backendBase}${v}`
              })
              setLinks(l)
              setProgress(100)
            } else {
              pushEvent(eventName, typeof payload === 'string' ? payload : JSON.stringify(payload))
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch (err: any) {
      setEvents(prev => [...prev, { id: `${Date.now()}-${prev.length}`, kind: 'error', message: err?.message || String(err) }])
    } finally {
      setSubmitting(false)
      controllerRef.current = null
    }
  }, [url, description, audioBlob])

  useEffect(() => {
    if (videoUrl && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [videoUrl])

  const closeModal = useCallback(() => {
    setShowModal(false)
  }, [])

  const onCancel = useCallback(() => {
    controllerRef.current?.abort()
  }, [])

  return (
    <div className="container">
      <header>
        <h1>Demo Video Creator</h1>
        <p>Provide a URL, a short description, and a quick audio clip.</p>
      </header>

      <form onSubmit={onSubmit}>
        <div className="form-group">
          <label>Website URL</label>
          <input type="text" placeholder="github.com or https://github.com" value={url} onChange={e => setUrl(e.target.value)} required />
        </div>

        <div className="form-group">
          <label>Task Description</label>
          <textarea rows={3} placeholder="Describe what the agent should do..." value={description} onChange={e => setDescription(e.target.value)} />
        </div>

        <div className="form-group">
          <label>Voice Recording</label>
          <div className="recording-section">
            <AudioRecorder onRecorded={onRecorded} />
            <div className="hint">Allow microphone access and record a few sentences (10–20 seconds) for voice cloning.</div>
          </div>
        </div>

        <div className="actions">
          <button type="submit" disabled={!audioBlob || submitting}>
            {submitting ? 'Creating Video...' : 'Create Demo Video'}
          </button>
          {submitting && (
            <button type="button" onClick={onCancel} className="secondary">Cancel</button>
          )}
        </div>
      </form>

      <section className="stream">
        <h2>Progress</h2>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <div className="events">
          {events.map(ev => (
            <div key={ev.id} className={`event ${ev.kind === 'error' ? 'error' : ev.kind === 'done' ? 'done' : ''}`}>
              {ev.kind.toUpperCase()}: {ev.message}
            </div>
          ))}
        </div>
      </section>

      {showModal && (
        <div className="modal-overlay" role="dialog" aria-modal="true">
          <div className="modal">
            <div className="modal-header">
              <h3>{videoUrl ? 'Your Final Video' : 'Generating your video...'}</h3>
              <button type="button" className="modal-close" onClick={closeModal}>×</button>
            </div>
            <div className="modal-body">
              {!videoUrl && (
                <>
                  <div className="hint" style={{ marginBottom: 8 }}>Now, just wait for your video! Play snake to pass time!</div>
                  <div className="game-wrapper">
                    <Snake />
                  </div>
                  <button type="button" className="secondary" onClick={() => setShowLogs(v => !v)} style={{ marginTop: 12 }}>
                    {showLogs ? 'Hide Logs' : 'Show Logs'}
                  </button>
                  {showLogs && (
                    <div className="events" style={{ maxHeight: 240, marginTop: 10 }}>
                      {events.map(ev => (
                        <div key={ev.id} className={`event ${ev.kind === 'error' ? 'error' : ev.kind === 'done' ? 'done' : ''}`}>
                          {ev.kind.toUpperCase()}: {ev.message}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}

              {videoUrl && (
                <div ref={resultRef}>
                  <video src={videoUrl} controls />
                </div>
              )}
            </div>
            <div className="modal-footer">
              {!videoUrl ? (
                <button type="button" className="secondary" onClick={onCancel}>Cancel</button>
              ) : (
                <>
                  <a href={videoUrl} download={`demo_voiceover_${Date.now()}.mp4`} className="secondary" style={{ textDecoration: 'none', padding: '10px 14px', borderRadius: 10 }}>Download MP4</a>
                  <a href={videoUrl} target="_blank" rel="noreferrer" className="secondary" style={{ textDecoration: 'none', padding: '10px 14px', borderRadius: 10 }}>Open in new tab</a>
                  <button type="button" className="secondary" onClick={() => navigator.clipboard.writeText(videoUrl)}>Copy link</button>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {videoUrl && (
        <section className="result" ref={resultRef}>
          <h2>Your Final Video</h2>
          <video src={videoUrl} controls />
          <div className="actions" style={{ marginTop: 10 }}>
            <a href={videoUrl} download className="secondary" style={{ textDecoration: 'none', padding: '8px 12px', borderRadius: 10, background: '#26324f', color: 'white' }}>Download MP4</a>
            <a href={videoUrl} target="_blank" rel="noreferrer" className="secondary" style={{ textDecoration: 'none', padding: '8px 12px', borderRadius: 10, background: '#26324f', color: 'white' }}>Open in new tab</a>
            <button type="button" className="secondary" onClick={() => navigator.clipboard.writeText(videoUrl)}>Copy link</button>
          </div>
          <div className="links" style={{ marginTop: 8 }}>
            {Object.entries(links).map(([k, v]) => (
              <a key={k} href={v} target="_blank" rel="noreferrer">{k.replace('_',' ')}</a>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}


