import React, { useCallback, useEffect, useRef, useState } from 'react'

type Props = {
  onRecorded: (blob: Blob | null) => void
}

export default function AudioRecorder({ onRecorded }: Props) {
  const [recording, setRecording] = useState(false)
  const [url, setUrl] = useState<string | null>(null)
  const [seconds, setSeconds] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const drawReqRef = useRef<number | null>(null)
  const dataArrayRef = useRef<Uint8Array | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const startedAtRef = useRef<number | null>(null)
  
  // Sample script for voice recording (about 10 seconds to read)
  const RECORDING_SCRIPT = [
    "Hello,", "my", "name", "is", "[Your", "Name]", "and", "I'm", "excited", "to", "show", "you", 
    "this", "amazing", "demo.", "This", "voice", "will", "be", "used", "to", "create", 
    "professional", "product", "demonstrations", "with", "AI", "assistance."
  ]

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    if (drawReqRef.current) cancelAnimationFrame(drawReqRef.current)
    audioCtxRef.current?.close().catch(() => {})
  }, [])

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    streamRef.current = stream
    chunksRef.current = []
    const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' })
    mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      setUrl(URL.createObjectURL(blob))
      onRecorded(blob)
      if (drawReqRef.current) cancelAnimationFrame(drawReqRef.current)
      audioCtxRef.current?.close().catch(() => {})
      audioCtxRef.current = null
      analyserRef.current = null
      dataArrayRef.current = null
    }
    mediaRecorderRef.current = mr
    mr.start()
    setRecording(true)
    startedAtRef.current = Date.now()
    setSeconds(0)

    // WebAudio visualization
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)()
    audioCtxRef.current = ctx
    const source = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 2048
    analyser.smoothingTimeConstant = 0.85
    source.connect(analyser)
    analyserRef.current = analyser
    dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount)

    const draw = () => {
      const canvas = canvasRef.current
      const analyserNode = analyserRef.current
      const dataArr = dataArrayRef.current
      if (!canvas || !analyserNode || !dataArr) return
      const c = canvas.getContext('2d')!
      const { width, height } = canvas
      c.clearRect(0, 0, width, height)

      analyserNode.getByteTimeDomainData(dataArr)
      // background gradient
      const grad = c.createLinearGradient(0, 0, width, height)
      grad.addColorStop(0, 'rgba(106,163,255,0.15)')
      grad.addColorStop(1, 'rgba(77,214,182,0.12)')
      c.fillStyle = grad
      c.fillRect(0, 0, width, height)

      // waveform - frequency bars style
      c.fillStyle = '#3b82f6'
      const barWidth = width / 64
      const barSpacing = 2
      
      // Get frequency data instead
      analyserNode.getByteFrequencyData(dataArr)
      
      for (let i = 0; i < 64; i++) {
        const barHeight = (dataArr[i] / 255) * height * 0.8
        const x = i * (barWidth + barSpacing)
        const y = height - barHeight
        
        // Create gradient for each bar
        const gradient = c.createLinearGradient(0, height, 0, y)
        gradient.addColorStop(0, '#3b82f6')
        gradient.addColorStop(0.5, '#60a5fa')
        gradient.addColorStop(1, '#93c5fd')
        
        c.fillStyle = gradient
        c.fillRect(x, y, barWidth, barHeight)
      }

      // update timer
      if (startedAtRef.current) setSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000))

      drawReqRef.current = requestAnimationFrame(draw)
    }
    draw()
  }, [onRecorded])

  const stop = useCallback(() => {
    mediaRecorderRef.current?.stop()
    streamRef.current?.getTracks().forEach(t => t.stop())
    setRecording(false)
  }, [])

  const reset = useCallback(() => {
    setUrl(null)
    onRecorded(null)
  }, [onRecorded])

  return (
    <div className="recorder">
      {!url && (
        <div className="script-prompt">
          <div className="script-text">
            {RECORDING_SCRIPT.map((word, index) => (
              <span key={index} className="word">{word}</span>
            ))}
          </div>
          <div className="prompt-hint">Read the text above naturally when recording</div>
        </div>
      )}
      
      <div className="wave-shell">
        <canvas ref={canvasRef} className="wave" width={720} height={120} />
        <div className="timer">{Math.floor(seconds / 60).toString().padStart(2, '0')}:{(seconds % 60).toString().padStart(2, '0')}</div>
        
        {/* Record button overlay */}
        <div className="record-overlay">
          <button 
            type="button" 
            className={`record-btn ${recording ? 'recording' : ''}`}
            onClick={recording ? stop : start}
          >
            <div className="record-icon">
              {recording ? (
                <div className="stop-icon" />
              ) : (
                <div className="mic-icon" />
              )}
            </div>
          </button>
        </div>
      </div>
      
      {url && (
        <div className="playback-section">
          <div className="playback-info">
            <span className="playback-label">Recording ready</span>
            <button type="button" className="reset-btn" onClick={reset}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
                <path d="M21 3v5h-5"/>
                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
                <path d="M3 21v-5h5"/>
              </svg>
              Reset
            </button>
          </div>
          <audio src={url} controls className="audio-player" />
        </div>
      )}
    </div>
  )
}


