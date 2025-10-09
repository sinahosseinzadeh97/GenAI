import { Button, Typography, Box, Stack, Chip, Divider, IconButton, Tooltip } from '@mui/material'
import { ContentCopy, CheckCircle, Person, WorkOutline, AlternateEmail } from '@mui/icons-material'
import { useState } from 'react'

type EmailItem = {
  company: string;
  contact: {
    full_name: string;
    title: string;
    email: string;
  } | string;
  subject: string;
  body: string;
}

export default function EmailsList({ items }: { items: EmailItem[] }) {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)

  if (!items?.length) return <Typography>No emails generated</Typography>
  
  const copy = async (txt: string, index: number) => { 
    try { 
      await navigator.clipboard.writeText(txt) 
      setCopiedIndex(index)
      setTimeout(() => setCopiedIndex(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }
  
  return (
    <Stack spacing={3}>
      {items.map((e, i) => (
        <Box 
          key={i} 
          sx={{ 
            border: '2px solid transparent',
            background: 'linear-gradient(white, white) padding-box, linear-gradient(135deg, #667eea, #764ba2) border-box',
            borderRadius: 3,
            p: 3,
            transition: 'all 0.3s ease',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 8px 24px rgba(102, 126, 234, 0.15)'
            }
          }}
        >
          <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
            <Stack direction="row" alignItems="center" spacing={1}>
              <Chip 
                label={`#${i + 1}`} 
                size="small" 
                sx={{ 
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  fontWeight: 'bold'
                }} 
              />
              <Typography variant="h6" fontWeight="bold">
                {e.company}
              </Typography>
            </Stack>
            <Tooltip title={copiedIndex === i ? "Copied!" : "Copy email"}>
              <IconButton 
                onClick={() => copy(`Subject: ${e.subject}\n\n${e.body}`, i)}
                sx={{
                  background: copiedIndex === i 
                    ? 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)'
                    : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  '&:hover': {
                    background: copiedIndex === i
                      ? 'linear-gradient(135deg, #45a049 0%, #4caf50 100%)'
                      : 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)',
                  }
                }}
              >
                {copiedIndex === i ? <CheckCircle /> : <ContentCopy />}
              </IconButton>
            </Tooltip>
          </Stack>
          
          {typeof e.contact !== 'string' && (
            <Stack spacing={0.5} mb={2} sx={{ 
              p: 2, 
              borderRadius: 2,
              background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)'
            }}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Person sx={{ fontSize: 18, color: '#667eea' }} />
                <Typography variant="body2" fontWeight="600">
                  {e.contact.full_name}
                </Typography>
              </Stack>
              <Stack direction="row" alignItems="center" spacing={1}>
                <WorkOutline sx={{ fontSize: 18, color: '#764ba2' }} />
                <Typography variant="body2" color="text.secondary">
                  {e.contact.title}
                </Typography>
              </Stack>
              <Stack direction="row" alignItems="center" spacing={1}>
                <AlternateEmail sx={{ fontSize: 18, color: '#667eea' }} />
                <Typography variant="body2" color="text.secondary">
                  {e.contact.email}
                </Typography>
              </Stack>
            </Stack>
          )}
          
          <Divider sx={{ my: 2 }} />
          
          <Stack spacing={2}>
            <Box>
              <Typography variant="caption" color="text.secondary" fontWeight="600">
                SUBJECT
              </Typography>
              <Typography variant="body1" fontWeight="600" sx={{ mt: 0.5 }}>
                {e.subject}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="caption" color="text.secondary" fontWeight="600">
                MESSAGE
              </Typography>
              <Typography 
                variant="body1" 
                sx={{ 
                  mt: 1,
                  whiteSpace: 'pre-wrap',
                  lineHeight: 1.8,
                  p: 2,
                  borderRadius: 2,
                  background: '#f8f9fa'
                }}
              >
                {e.body}
              </Typography>
            </Box>
          </Stack>
        </Box>
      ))}
    </Stack>
  )
}
