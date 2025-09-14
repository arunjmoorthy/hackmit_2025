import React, { useEffect, useRef, useState } from 'react'

export default function Snake() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const reqRef = useRef<number | null>(null)
  const [score, setScore] = useState(0)
  const [high, setHigh] = useState<number>(() => Number(localStorage.getItem('snake_highscore') || 0))
  const [paused, setPaused] = useState(false)
  const [speedTicks, setSpeedTicks] = useState(12) // higher is slower
  const [started, setStarted] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')!

    const grid = 16
    let count = 0

    const snake = {
      x: 160,
      y: 160,
      dx: grid,
      dy: 0,
      cells: [] as Array<{ x: number; y: number }>,
      maxCells: 4,
    }

    const apple = { x: 320, y: 320 }

    function getRandomInt(min: number, max: number) {
      return Math.floor(Math.random() * (max - min)) + min
    }

    function reset() {
      setHigh(h => {
        const nh = Math.max(h, score)
        localStorage.setItem('snake_highscore', String(nh))
        return nh
      })
      setScore(0)
      setSpeedTicks(12)
      snake.x = 160
      snake.y = 160
      snake.cells = []
      snake.maxCells = 4
      snake.dx = grid
      snake.dy = 0
      apple.x = getRandomInt(0, 25) * grid
      apple.y = getRandomInt(0, 25) * grid
    }

    function loop() {
      reqRef.current = requestAnimationFrame(loop)
      // Always draw; only move when started and not paused
      if (++count < speedTicks) { return }
      count = 0
      context.clearRect(0, 0, canvas.width, canvas.height)

      if (started && !paused) {
        snake.x += snake.dx
        snake.y += snake.dy
      }

      if (snake.x < 0) snake.x = canvas.width - grid
      else if (snake.x >= canvas.width) snake.x = 0
      if (snake.y < 0) snake.y = canvas.height - grid
      else if (snake.y >= canvas.height) snake.y = 0

      snake.cells.unshift({ x: snake.x, y: snake.y })
      if (snake.cells.length > snake.maxCells) snake.cells.pop()

      // apple
      context.fillStyle = '#ef4444'
      context.fillRect(apple.x, apple.y, grid - 1, grid - 1)

      // snake
      context.fillStyle = '#22c55e'
      snake.cells.forEach((cell, index) => {
        context.fillRect(cell.x, cell.y, grid - 1, grid - 1)
        if (cell.x === apple.x && cell.y === apple.y) {
          snake.maxCells++
          setScore(s => {
            const ns = s + 1
            // every 5 points, slightly increase speed
            if (ns % 5 === 0 && speedTicks > 6) setSpeedTicks(t => Math.max(6, t - 1))
            return ns
          })
          apple.x = getRandomInt(0, 25) * grid
          apple.y = getRandomInt(0, 25) * grid
        }
        for (let i = index + 1; i < snake.cells.length; i++) {
          if (cell.x === snake.cells[i].x && cell.y === snake.cells[i].y) {
            reset()
          }
        }
      })

      // HUD
      context.fillStyle = '#9ca3af'
      context.font = '12px system-ui'
      context.fillText(`Score: ${score}  High: ${Math.max(high, score)}`, 8, 14)
      if (paused) {
        context.fillStyle = '#ffffff88'
        context.fillRect(0, 0, canvas.width, canvas.height)
        context.fillStyle = '#111827'
        context.font = 'bold 16px system-ui'
        context.fillText('Paused', canvas.width / 2 - 26, canvas.height / 2)
      }
      if (!started) {
        context.fillStyle = '#ffffff88'
        context.fillRect(0, 0, canvas.width, canvas.height)
        context.fillStyle = '#111827'
        context.font = 'bold 14px system-ui'
        context.fillText('Press Start to play Snake', 110, canvas.height / 2)
      }
    }

    function keyHandler(e: KeyboardEvent) {
      if (!started) return
      if (e.key === ' ') { setPaused(p => !p); e.preventDefault(); return }
      if (paused) return
      if (e.key === 'ArrowLeft' && snake.dx === 0) { snake.dx = -grid; snake.dy = 0 }
      else if (e.key === 'ArrowUp' && snake.dy === 0) { snake.dy = -grid; snake.dx = 0 }
      else if (e.key === 'ArrowRight' && snake.dx === 0) { snake.dx = grid; snake.dy = 0 }
      else if (e.key === 'ArrowDown' && snake.dy === 0) { snake.dy = grid; snake.dx = 0 }
    }
    window.addEventListener('keydown', keyHandler)
    reqRef.current = requestAnimationFrame(loop)

    return () => {
      if (reqRef.current) cancelAnimationFrame(reqRef.current)
      window.removeEventListener('keydown', keyHandler)
    }
  }, [started, paused, speedTicks, score, high])

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 8 }}>
      <canvas ref={canvasRef} width={400} height={400} className="snake-canvas" />
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="secondary" type="button" onClick={() => { setStarted(true); setPaused(false) }}>Start</button>
        <button className="secondary" type="button" onClick={() => setStarted(false)}>Stop</button>
        <button className="secondary" type="button" onClick={() => setPaused(p => !p)} disabled={!started}>{paused ? 'Resume (Space)' : 'Pause (Space)'}</button>
      </div>
    </div>
  )
}


