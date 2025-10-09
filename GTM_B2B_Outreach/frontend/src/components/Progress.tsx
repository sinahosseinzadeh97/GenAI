import { useEffect, useState, useRef } from 'react'

export default function Progress({ taskId, onDone }: { taskId: string; onDone: () => void }) {
  const [pct, setPct] = useState(0)
  const pollingRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    console.log('[Progress] Starting polling for task:', taskId)
    let completed = false
    
    // Poll progress every 500ms
    const poll = async () => {
      try {
        const res = await fetch(`/api/result/${taskId}`)
        if (!res.ok) {
          console.log('[Progress] Result not ready yet, continuing polling...')
          return
        }
        
        const data = await res.json()
        
        // Estimate progress based on data availability
        let progress = 5
        if (data.companies && data.companies.length > 0) progress = 30
        if (data.contacts && data.contacts.length > 0) progress = 55
        if (data.research && data.research.length > 0) progress = 80
        if (data.emails && data.emails.length > 0) progress = 100
        
        console.log('[Progress] Progress:', progress, '- Data:', data)
        setPct(progress)
        
        if (progress >= 100 && !completed) {
          completed = true
          console.log('[Progress] Complete! Calling onDone...')
          if (pollingRef.current) clearInterval(pollingRef.current)
          onDone()
        }
      } catch (err) {
        console.error('[Progress] Polling error:', err)
      }
    }
    
    // Initial poll
    poll()
    
    // Set up interval
    pollingRef.current = setInterval(poll, 500)
    
    return () => {
      console.log('[Progress] Cleanup - stopping polling')
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [taskId, onDone])

  return (
    <div>
      <div style={{ 
        width: '100%', 
        height: 12,
        background: 'linear-gradient(90deg, #f0f0f0 0%, #e0e0e0 100%)', 
        borderRadius: 12,
        overflow: 'hidden',
        position: 'relative'
      }}>
        <div style={{ 
          height: '100%', 
          width: `${pct}%`, 
          borderRadius: 12, 
          background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
          transition: 'width 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          boxShadow: pct > 0 ? '0 0 10px rgba(102, 126, 234, 0.5)' : 'none',
          position: 'relative',
          overflow: 'hidden'
        }}>
          {pct > 0 && (
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
              animation: 'shimmer 2s infinite'
            }} />
          )}
        </div>
      </div>
      <div style={{ 
        textAlign: 'center', 
        marginTop: 12, 
        fontSize: 18,
        fontWeight: 'bold',
        color: '#667eea'
      }}>
        {pct}%
      </div>
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  )
}
