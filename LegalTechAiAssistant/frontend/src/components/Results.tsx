import { Card, CardContent, Typography, List, ListItem, ListItemText, Divider } from '@mui/material'
import React from 'react'
import { Source } from '../types'

export default function Results({ answer, sources, laws, draft }: { answer?: string, sources?: Source[], laws?: Source[], draft?: string }) {
  return (
    <Card sx={{ borderRadius: 4, p: 2, boxShadow: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>Risultati</Typography>
        {answer && <>
          <Typography variant="subtitle1">Risposta</Typography>
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', mb: 2 }}>{answer}</Typography>
        </>}
        {draft && <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1">Bozza</Typography>
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>{draft}</Typography>
        </>}
        {sources && sources.length>0 && <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1">Fonti documento</Typography>
          <List dense>
            {sources.map((s, i)=> (
              <ListItem key={i}>
                <ListItemText primary={s.title || `Documento ${s.document_id}`} secondary={s.chunk.slice(0, 180) + '…'} />
              </ListItem>
            ))}
          </List>
        </>}
        {laws && laws.length>0 && <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1">Riferimenti normativi</Typography>
          <List dense>
            {laws.map((s, i)=> (
              <ListItem key={i}>
                <ListItemText primary={s.title} secondary={s.chunk.slice(0, 180) + '…'} />
              </ListItem>
            ))}
          </List>
        </>}
      </CardContent>
    </Card>
  )
}