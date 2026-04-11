import React from 'react'
import { Container, Grid, Typography, Alert } from '@mui/material'
import UploadCard from './components/UploadCard'
import ChatCard from './components/ChatCard'
import Results from './components/Results'
import { askQuestion, uploadDocument } from './api'

export default function App(){
  const [answer, setAnswer] = React.useState<string>()
  const [sources, setSources] = React.useState<any[]>()
  const [laws, setLaws] = React.useState<any[]>()
  const [draft, setDraft] = React.useState<string>()
  const [msg, setMsg] = React.useState<string>()

  async function handleUpload(f: File){
    setMsg('Caricamento…')
    try{
      const res = await uploadDocument(f)
      setMsg(`Documento #${res.document_id} caricato. Workflow #${res.workflow_id} avviato.`)
    }catch(e:any){ setMsg(e.message) }
  }

  async function handleAsk(q: string, action?: string){
    setMsg('Interrogo il RAG…')
    try{
      const res = await askQuestion(q, action)
      setAnswer(res.answer); setSources(res.sources); setLaws(res.laws); setDraft(res.draft)
      setMsg('')
    }catch(e:any){ setMsg(e.message) }
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>⚖️ LegalTech AI Assistant — MVP</Typography>
      {msg && <Alert severity="info" sx={{ mb: 2 }}>{msg}</Alert>}
      <Grid container spacing={3}>
        <Grid item xs={12}><UploadCard onUpload={handleUpload} /></Grid>
        <Grid item xs={12}><ChatCard onAsk={handleAsk} /></Grid>
        <Grid item xs={12}><Results answer={answer} sources={sources} laws={laws} draft={draft} /></Grid>
      </Grid>
    </Container>
  )
}