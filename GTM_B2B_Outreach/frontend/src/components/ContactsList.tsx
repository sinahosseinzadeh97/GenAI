import { Typography, Box, Stack, Chip, Avatar } from '@mui/material'
import { Person, WorkOutline, AlternateEmail, CheckCircle } from '@mui/icons-material'
import type { CompanyContacts } from '../types'

export default function ContactsList({ data }: { data: CompanyContacts[] }) {
  if (!data?.length) return <Typography>No contacts found</Typography>
  
  return (
    <Stack spacing={2}>
      {data.map((c) => (
        <Box 
          key={c.name}
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
          <Typography variant="h6" fontWeight="bold" mb={2}>
            {c.name}
          </Typography>
          <Stack spacing={1.5}>
            {(c.contacts || []).slice(0, 3).map((p, i) => (
              <Box 
                key={i}
                sx={{
                  p: 2,
                  borderRadius: 2,
                  background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2
                }}
              >
                <Avatar 
                  sx={{ 
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    width: 40,
                    height: 40
                  }}
                >
                  <Person />
                </Avatar>
                <Box sx={{ flex: 1 }}>
                  <Stack direction="row" alignItems="center" spacing={1} mb={0.5}>
                    <Typography variant="body1" fontWeight="600">
                      {p.full_name}
                    </Typography>
                    {p.inferred && (
                      <Chip 
                        label="inferred" 
                        size="small" 
                        variant="outlined"
                        sx={{ 
                          height: 20,
                          fontSize: 10,
                          color: '#764ba2',
                          borderColor: '#764ba2'
                        }} 
                      />
                    )}
                  </Stack>
                  <Stack direction="row" alignItems="center" spacing={0.5} mb={0.5}>
                    <WorkOutline sx={{ fontSize: 14, color: '#764ba2' }} />
                    <Typography variant="body2" color="text.secondary">
                      {p.title}
                    </Typography>
                  </Stack>
                  <Stack direction="row" alignItems="center" spacing={0.5}>
                    <AlternateEmail sx={{ fontSize: 14, color: '#667eea' }} />
                    <Typography variant="body2" color="text.secondary">
                      {p.email}
                    </Typography>
                  </Stack>
                </Box>
              </Box>
            ))}
          </Stack>
        </Box>
      ))}
    </Stack>
  )
}
