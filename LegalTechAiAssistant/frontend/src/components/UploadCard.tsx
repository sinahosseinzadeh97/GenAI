import { Card, CardContent, Typography, Button } from '@mui/material'
import React from 'react'

export default function UploadCard({ onUpload }: { onUpload: (f: File) => void }) {
  const inputRef = React.useRef<HTMLInputElement>(null)
  const [fileName, setFileName] = React.useState<string>('')

  return (
    <Card sx={{ borderRadius: 4, p: 2, boxShadow: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>Carica documento</Typography>
        <input type="file" ref={inputRef} style={{ display: 'none' }} onChange={e => {
          const f = e.target.files?.[0]
          if (f) { setFileName(f.name); onUpload(f) }
        }} />
        <Button variant="contained" onClick={() => inputRef.current?.click()}>Seleziona file</Button>
        <Typography variant="body2" sx={{ mt: 1, opacity: .8 }}>{fileName || 'PDF, DOCX, TXT'}</Typography>
      </CardContent>
    </Card>
  )
}