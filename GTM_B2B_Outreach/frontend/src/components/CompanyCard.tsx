import { Typography, Box, Chip, Stack } from '@mui/material'
import { Language, TrendingUp } from '@mui/icons-material'
import type { Company } from '../types'

export default function CompanyCard({ c, idx }: { c: Company; idx: number }) {
  return (
    <Box
      sx={{
        border: '2px solid transparent',
        background: 'linear-gradient(white, white) padding-box, linear-gradient(135deg, #667eea, #764ba2) border-box',
        borderRadius: 3,
        p: 2.5,
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: '0 8px 24px rgba(102, 126, 234, 0.2)'
        }
      }}
    >
      <Stack direction="row" alignItems="center" spacing={1} mb={1}>
        <Chip 
          label={`#${idx}`} 
          size="small" 
          sx={{ 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            fontWeight: 'bold'
          }} 
        />
        <Typography variant="h6" fontWeight="bold">
          {c.name}
        </Typography>
      </Stack>
      
      <Stack direction="row" alignItems="center" spacing={0.5} mb={1.5}>
        <Language sx={{ fontSize: 16, color: '#667eea' }} />
        <Typography 
          variant="body2" 
          component="a" 
          href={c.website} 
          target="_blank"
          sx={{ 
            color: '#667eea',
            textDecoration: 'none',
            '&:hover': { textDecoration: 'underline' }
          }}
        >
          {c.website}
        </Typography>
      </Stack>
      
      <Stack direction="row" spacing={1} alignItems="flex-start">
        <TrendingUp sx={{ fontSize: 20, color: '#764ba2', mt: 0.5 }} />
        <Typography variant="body1" sx={{ lineHeight: 1.6 }}>
          {c.why_fit}
        </Typography>
      </Stack>
    </Box>
  )
}
