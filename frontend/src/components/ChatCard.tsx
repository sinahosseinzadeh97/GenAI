import { Card, CardContent, Typography, TextField, Button, Stack, MenuItem } from '@mui/material'
import React from 'react'

export default function ChatCard({ onAsk }: { onAsk: (q: string, action?: string) => void }) {
  const [q, setQ] = React.useState('Quali clausole essenziali?')
  const [action, setAction] = React.useState<string>('')

  return (
    <Card sx={{ borderRadius: 4, p: 2, boxShadow: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>Assistente legale (RAG)</Typography>
        <Stack spacing={2}>
          <TextField fullWidth label="Domanda" value={q} onChange={e=>setQ(e.target.value)} />
          <TextField select fullWidth label="Azione opzionale" value={action} onChange={e=>setAction(e.target.value)}>
            <MenuItem value="">Nessuna</MenuItem>
            <MenuItem value="client_letter">Email al cliente</MenuItem>
            <MenuItem value="case_summary">Sommario del caso</MenuItem>
            <MenuItem value="contract">Bozza clausola contratto</MenuItem>
          </TextField>
          <Button variant="contained" onClick={()=>onAsk(q, action || undefined)}>Chiedi</Button>
        </Stack>
      </CardContent>
    </Card>
  )
}