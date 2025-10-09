import { Typography, Box, Stack, Chip } from '@mui/material'
import { Lightbulb } from '@mui/icons-material'
import type { CompanyResearch } from '../types'

export default function ResearchList({ data }: { data: CompanyResearch[] }) {
  if (!data?.length) return <Typography>No research insights</Typography>
  
  return (
    <Stack spacing={2}>
      {data.map((r) => (
        <Box 
          key={r.name}
          sx={{
            border: '2px solid transparent',
            background: 'linear-gradient(white, white) padding-box, linear-gradient(135deg, #667eea, #764ba2) border-box',
            borderRadius: 3,
            p: 2.5,
            transition: 'all 0.3s ease',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 8px 24px rgba(102, 126, 234, 0.15)'
            }
          }}
        >
          <Stack direction="row" alignItems="center" spacing={1} mb={2}>
            <Typography variant="h6" fontWeight="bold">
              {r.name}
            </Typography>
            <Chip 
              label={`${(r.insights || []).length} insights`}
              size="small"
              sx={{
                background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                color: '#667eea',
                fontWeight: 600
              }}
            />
          </Stack>
          <Stack spacing={1.5}>
            {(r.insights || []).slice(0, 4).map((t, i) => (
              <Box 
                key={i}
                sx={{
                  p: 2,
                  borderRadius: 2,
                  background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)',
                  display: 'flex',
                  gap: 1.5,
                  alignItems: 'flex-start'
                }}
              >
                <Lightbulb sx={{ fontSize: 20, color: '#667eea', mt: 0.2 }} />
                <Typography variant="body2" sx={{ lineHeight: 1.6, flex: 1 }}>
                  {t}
                </Typography>
              </Box>
            ))}
          </Stack>
        </Box>
      ))}
    </Stack>
  )
}
