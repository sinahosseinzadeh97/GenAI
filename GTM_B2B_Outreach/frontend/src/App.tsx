import { useState } from 'react'
import { Button, TextField, MenuItem, Typography, Box, Paper, Fade, Container, Chip, Stack, Dialog, DialogTitle, DialogContent, IconButton, Divider } from '@mui/material'
import { RocketLaunch, Business, Email, Insights, ContactMail, Info, Close, AutoAwesome, Search, Psychology, TrendingUp } from '@mui/icons-material'
import Progress from './components/Progress'
import CompanyCard from './components/CompanyCard'
import ContactsList from './components/ContactsList'
import ResearchList from './components/ResearchList'
import EmailsList from './components/EmailsList'
import { startRun, fetchResult } from './api'
import type { RunPayload, PipelineResult } from './types'

export default function App() {
  const [form, setForm] = useState<RunPayload>({
    target_desc: '',
    offering_desc: '',
    sender_name: 'Sales Team',
    sender_company: 'Our Company',
    calendar_link: '',
    num_companies: 5,
    email_style: 'Professional'
  })
  const [taskId, setTaskId] = useState<string>()
  const [results, setResults] = useState<PipelineResult | null>(null)
  const [aboutOpen, setAboutOpen] = useState(false)

  const start = async () => {
    console.log('[App] Starting outreach with form:', form)
    const { task_id } = await startRun(form)
    console.log('[App] Got task_id:', task_id)
    setTaskId(task_id)
    setResults(null)
  }

  const done = async () => {
    console.log('[App] done() called with taskId:', taskId)
    if (!taskId) {
      console.error('[App] No taskId found!')
      return
    }
    console.log('[App] Fetching result...')
    const res = await fetchResult(taskId)
    console.log('[App] Got result:', res)
    setResults(res)
  }

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      py: 4
    }}>
      <Container maxWidth="lg">
        {/* Header */}
        <Fade in timeout={800}>
          <Paper 
            elevation={0} 
            sx={{ 
              p: 4, 
              mb: 3, 
              borderRadius: 4,
              background: 'rgba(255,255,255,0.95)',
              backdropFilter: 'blur(10px)'
            }}
          >
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Box>
                <Stack direction="row" spacing={2} alignItems="center" mb={1}>
                  <RocketLaunch sx={{ fontSize: 40, color: '#667eea' }} />
                  <Typography variant="h3" fontWeight="bold" sx={{ 
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                  }}>
                    GTM B2B Outreach
                  </Typography>
                </Stack>
                <Typography variant="body1" color="text.secondary" sx={{ ml: 7 }}>
                  🎯 Find companies → 👥 contacts → 🔍 research → ✉️ generate personalized emails
                </Typography>
              </Box>
              <Button
                variant="outlined"
                startIcon={<Info />}
                onClick={() => setAboutOpen(true)}
                sx={{
                  borderColor: '#667eea',
                  color: '#667eea',
                  '&:hover': {
                    borderColor: '#764ba2',
                    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)'
                  }
                }}
              >
                About
              </Button>
            </Stack>
          </Paper>
        </Fade>

        {/* About Modal */}
        <Dialog 
          open={aboutOpen} 
          onClose={() => setAboutOpen(false)}
          maxWidth="md"
          fullWidth
          PaperProps={{
            sx: {
              borderRadius: 4,
              background: 'linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.98) 100%)',
              backdropFilter: 'blur(10px)'
            }
          }}
        >
          <DialogTitle sx={{ 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            py: 2
          }}>
            <Stack direction="row" alignItems="center" spacing={1.5}>
              <AutoAwesome sx={{ fontSize: 28 }} />
              <Typography variant="h5" fontWeight="bold">
                About GTM B2B Outreach
              </Typography>
            </Stack>
            <IconButton onClick={() => setAboutOpen(false)} sx={{ color: 'white' }}>
              <Close />
            </IconButton>
          </DialogTitle>
          
          <DialogContent sx={{ p: 4 }}>
            <Stack spacing={3}>
              {/* Main Description */}
              <Box>
                <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ 
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  🚀 What is this?
                </Typography>
                <Typography variant="body1" paragraph sx={{ lineHeight: 1.8 }}>
                  GTM B2B Outreach is an <strong>AI-powered platform</strong> that automates the entire cold email outreach process. 
                  It uses advanced AI agents (GPT-4o-mini) and web intelligence (Exa API) to find target companies, 
                  research them deeply, locate decision makers, and generate highly personalized emails.
                </Typography>
              </Box>

              <Divider />

              {/* How it Works */}
              <Box>
                <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ 
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  ⚙️ How it works?
                </Typography>
                <Stack spacing={2}>
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                    <Box sx={{ 
                      minWidth: 40, 
                      height: 40, 
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontWeight: 'bold'
                    }}>
                      1
                    </Box>
                    <Box>
                      <Stack direction="row" alignItems="center" spacing={1} mb={0.5}>
                        <Search sx={{ color: '#667eea', fontSize: 20 }} />
                        <Typography variant="subtitle1" fontWeight="600">Company Discovery</Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        AI agent searches the web using your targeting criteria (industry, location, size) and finds perfectly matching B2B companies.
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                    <Box sx={{ 
                      minWidth: 40, 
                      height: 40, 
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontWeight: 'bold'
                    }}>
                      2
                    </Box>
                    <Box>
                      <Stack direction="row" alignItems="center" spacing={1} mb={0.5}>
                        <ContactMail sx={{ color: '#667eea', fontSize: 20 }} />
                        <Typography variant="subtitle1" fontWeight="600">Contact Finding</Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        Identifies 2-3 key decision makers per company (founders, GTM leads, sales directors) with their emails, inferred when needed.
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                    <Box sx={{ 
                      minWidth: 40, 
                      height: 40, 
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontWeight: 'bold'
                    }}>
                      3
                    </Box>
                    <Box>
                      <Stack direction="row" alignItems="center" spacing={1} mb={0.5}>
                        <Psychology sx={{ color: '#667eea', fontSize: 20 }} />
                        <Typography variant="subtitle1" fontWeight="600">Deep Research</Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        Gathers insights from company websites, blogs, product pages, and Reddit discussions to understand their challenges and opportunities.
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                    <Box sx={{ 
                      minWidth: 40, 
                      height: 40, 
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontWeight: 'bold'
                    }}>
                      4
                    </Box>
                    <Box>
                      <Stack direction="row" alignItems="center" spacing={1} mb={0.5}>
                        <Email sx={{ color: '#667eea', fontSize: 20 }} />
                        <Typography variant="subtitle1" fontWeight="600">Personalized Emails</Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        AI crafts unique, personalized cold emails referencing specific insights about each company, with customizable tone (Professional, Casual, Cold, Consultative).
                      </Typography>
                    </Box>
                  </Box>
                </Stack>
              </Box>

              <Divider />

              {/* Key Features */}
              <Box>
                <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ 
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  ✨ Key Features
                </Typography>
                <Stack spacing={1}>
                  <Typography variant="body2">• <strong>Multi-Agent AI System:</strong> Specialized agents for each task (finding, research, writing)</Typography>
                  <Typography variant="body2">• <strong>Web Intelligence:</strong> Powered by Exa API for deep web and Reddit research</Typography>
                  <Typography variant="body2">• <strong>Real Personalization:</strong> References actual company details and recent news</Typography>
                  <Typography variant="body2">• <strong>Email Style Options:</strong> Choose tone that fits your brand</Typography>
                  <Typography variant="body2">• <strong>One-Click Copy:</strong> Export emails ready to send</Typography>
                  <Typography variant="body2">• <strong>Calendar Integration:</strong> Add your booking link automatically</Typography>
                </Stack>
              </Box>

              <Divider />

              {/* Use Cases */}
              <Box>
                <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ 
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  🎯 Perfect For
                </Typography>
                <Stack spacing={1}>
                  <Typography variant="body2">• <strong>SDRs & Sales Teams:</strong> Scale outbound without sacrificing quality</Typography>
                  <Typography variant="body2">• <strong>Founders & GTM Leaders:</strong> Launch campaigns in minutes</Typography>
                  <Typography variant="body2">• <strong>Agencies:</strong> Serve multiple clients efficiently</Typography>
                  <Typography variant="body2">• <strong>Marketers:</strong> ABM campaigns with deep personalization</Typography>
                </Stack>
              </Box>

              <Box sx={{ 
                p: 3, 
                borderRadius: 3,
                background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                border: '2px solid transparent',
                backgroundClip: 'padding-box'
              }}>
                <Stack direction="row" alignItems="center" spacing={1} mb={1}>
                  <TrendingUp sx={{ color: '#667eea' }} />
                  <Typography variant="subtitle1" fontWeight="bold">
                    Pro Tip
                  </Typography>
                </Stack>
                <Typography variant="body2" color="text.secondary">
                  For best results, be specific in your targeting criteria and offering description. 
                  The more detail you provide, the more relevant companies and personalized emails you'll get!
                </Typography>
              </Box>
            </Stack>
          </DialogContent>
        </Dialog>

        {/* Form */}
        <Fade in timeout={1000}>
          <Paper 
            elevation={0}
            sx={{ 
              p: 4, 
              mb: 3,
              borderRadius: 4,
              background: 'rgba(255,255,255,0.95)',
              backdropFilter: 'blur(10px)'
            }}
          >
            <Stack spacing={3}>
              <TextField 
                label="🎯 Target companies" 
                multiline 
                rows={3}
                value={form.target_desc} 
                onChange={e => setForm({ ...form, target_desc: e.target.value })}
                variant="outlined"
                fullWidth
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&:hover fieldset': { borderColor: '#667eea' },
                    '&.Mui-focused fieldset': { borderColor: '#667eea' }
                  }
                }}
              />
              <TextField 
                label="💼 Your offering" 
                multiline 
                rows={3}
                value={form.offering_desc} 
                onChange={e => setForm({ ...form, offering_desc: e.target.value })}
                variant="outlined"
                fullWidth
                sx={{
                  '& .MuiOutlinedInput-root': {
                    '&:hover fieldset': { borderColor: '#667eea' },
                    '&.Mui-focused fieldset': { borderColor: '#667eea' }
                  }
                }}
              />
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                <TextField 
                  label="👤 Your name" 
                  value={form.sender_name} 
                  onChange={e => setForm({ ...form, sender_name: e.target.value })}
                  fullWidth
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      '&:hover fieldset': { borderColor: '#667eea' },
                      '&.Mui-focused fieldset': { borderColor: '#667eea' }
                    }
                  }}
                />
                <TextField 
                  label="🏢 Your company" 
                  value={form.sender_company} 
                  onChange={e => setForm({ ...form, sender_company: e.target.value })}
                  fullWidth
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      '&:hover fieldset': { borderColor: '#667eea' },
                      '&.Mui-focused fieldset': { borderColor: '#667eea' }
                    }
                  }}
                />
              </Stack>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                <TextField 
                  label="📅 Calendar link (optional)" 
                  value={form.calendar_link} 
                  onChange={e => setForm({ ...form, calendar_link: e.target.value })}
                  fullWidth
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      '&:hover fieldset': { borderColor: '#667eea' },
                      '&.Mui-focused fieldset': { borderColor: '#667eea' }
                    }
                  }}
                />
                <TextField 
                  label="🔢 Num companies" 
                  type="number" 
                  value={form.num_companies} 
                  onChange={e => setForm({ ...form, num_companies: Number(e.target.value) })}
                  sx={{
                    minWidth: 150,
                    '& .MuiOutlinedInput-root': {
                      '&:hover fieldset': { borderColor: '#667eea' },
                      '&.Mui-focused fieldset': { borderColor: '#667eea' }
                    }
                  }}
                />
                <TextField 
                  label="✍️ Email style" 
                  select 
                  value={form.email_style} 
                  onChange={e => setForm({ ...form, email_style: e.target.value as any })}
                  sx={{
                    minWidth: 180,
                    '& .MuiOutlinedInput-root': {
                      '&:hover fieldset': { borderColor: '#667eea' },
                      '&.Mui-focused fieldset': { borderColor: '#667eea' }
                    }
                  }}
                >
                  {['Professional', 'Casual', 'Cold', 'Consultative'].map((s) => (
                    <MenuItem key={s} value={s}>{s}</MenuItem>
                  ))}
                </TextField>
              </Stack>
              <Button 
                variant="contained" 
                size="large"
                onClick={start} 
                disabled={!form.target_desc || !form.offering_desc}
                startIcon={<RocketLaunch />}
                sx={{
                  py: 1.5,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 10px 20px rgba(102, 126, 234, 0.3)'
                  },
                  transition: 'all 0.3s ease'
                }}
              >
                START OUTREACH
              </Button>
            </Stack>
          </Paper>
        </Fade>

        {/* Progress */}
        {taskId && (
          <Fade in timeout={500}>
            <Paper 
              elevation={0}
              sx={{ 
                p: 3, 
                mb: 3,
                borderRadius: 4,
                background: 'rgba(255,255,255,0.95)',
                backdropFilter: 'blur(10px)'
              }}
            >
              <Typography variant="h6" mb={2} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <RocketLaunch color="primary" /> Processing...
              </Typography>
              <Progress taskId={taskId} onDone={done} />
            </Paper>
          </Fade>
        )}

        {/* Results */}
        {results && (
          <>
            <Fade in timeout={800}>
              <Paper 
                elevation={0}
                sx={{ 
                  p: 4, 
                  mb: 3,
                  borderRadius: 4,
                  background: 'rgba(255,255,255,0.95)',
                  backdropFilter: 'blur(10px)'
                }}
              >
                <Stack direction="row" alignItems="center" spacing={1} mb={3}>
                  <Business sx={{ color: '#667eea', fontSize: 28 }} />
                  <Typography variant="h5" fontWeight="bold">Target Companies</Typography>
                  <Chip label={results.companies?.length || 0} color="primary" size="small" />
                </Stack>
                <Stack spacing={2}>
                  {results.companies?.map((c, i) => <CompanyCard key={i} c={c} idx={i + 1} />)}
                </Stack>
              </Paper>
            </Fade>

            <Fade in timeout={1000}>
              <Paper 
                elevation={0}
                sx={{ 
                  p: 4, 
                  mb: 3,
                  borderRadius: 4,
                  background: 'rgba(255,255,255,0.95)',
                  backdropFilter: 'blur(10px)'
                }}
              >
                <Stack direction="row" alignItems="center" spacing={1} mb={3}>
                  <ContactMail sx={{ color: '#667eea', fontSize: 28 }} />
                  <Typography variant="h5" fontWeight="bold">Contacts</Typography>
                  <Chip 
                    label={results.contacts?.reduce((acc, c) => acc + c.contacts.length, 0) || 0} 
                    color="primary" 
                    size="small" 
                  />
                </Stack>
                <ContactsList data={results.contacts} />
              </Paper>
            </Fade>

            <Fade in timeout={1200}>
              <Paper 
                elevation={0}
                sx={{ 
                  p: 4, 
                  mb: 3,
                  borderRadius: 4,
                  background: 'rgba(255,255,255,0.95)',
                  backdropFilter: 'blur(10px)'
                }}
              >
                <Stack direction="row" alignItems="center" spacing={1} mb={3}>
                  <Insights sx={{ color: '#667eea', fontSize: 28 }} />
                  <Typography variant="h5" fontWeight="bold">Research Insights</Typography>
                </Stack>
                <ResearchList data={results.research} />
              </Paper>
            </Fade>

            <Fade in timeout={1400}>
              <Paper 
                elevation={0}
                sx={{ 
                  p: 4, 
                  mb: 3,
                  borderRadius: 4,
                  background: 'rgba(255,255,255,0.95)',
                  backdropFilter: 'blur(10px)'
                }}
              >
                <Stack direction="row" alignItems="center" spacing={1} mb={3}>
                  <Email sx={{ color: '#667eea', fontSize: 28 }} />
                  <Typography variant="h5" fontWeight="bold">Personalized Emails</Typography>
                  <Chip label={results.emails?.length || 0} color="primary" size="small" />
                </Stack>
                <EmailsList items={results.emails} />
              </Paper>
            </Fade>
          </>
        )}
      </Container>
    </Box>
  )
}
