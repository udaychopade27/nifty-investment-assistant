# ETF Assistant Frontend

Modern React frontend for the ETF Investment Assistant system.

## Features

✅ **Real-time Dashboard** - View today's decisions, capital overview, and portfolio
✅ **Capital Management** - Set and manage monthly investment capital
✅ **Base Plan Generator** - Generate systematic investment plans
✅ **Tactical Signals** - View dip-based tactical investment opportunities
✅ **Portfolio Tracking** - Track all investments and P&L
✅ **Responsive Design** - Works on desktop, tablet, and mobile
✅ **Modern UI** - Built with Tailwind CSS and Lucide icons

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool & dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icon library
- **Axios** - HTTP client for API calls

## Setup Instructions

### Prerequisites

- Node.js 18+ installed
- ETF Assistant backend running on `http://localhost:8000`

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will open at `http://localhost:3000`

### Build for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── Dashboard.jsx      # Main dashboard component
│   ├── App.jsx            # Root App component
│   ├── main.jsx           # Entry point
│   └── index.css          # Global styles with Tailwind
├── index.html             # HTML template
├── package.json           # Dependencies
├── vite.config.js         # Vite configuration
├── tailwind.config.js     # Tailwind configuration
└── postcss.config.js      # PostCSS configuration
```

## API Integration

The frontend connects to the backend API at `http://localhost:8000`.

Vite proxy configuration handles API requests:
```javascript
// vite.config.js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  }
}
```

### API Endpoints Used

- `GET /api/v1/capital/current` - Get monthly capital
- `POST /api/v1/capital/set` - Set monthly capital
- `POST /api/v1/capital/generate-base-plan` - Generate base plan
- `GET /api/v1/decision/today` - Get today's decision
- `GET /api/v1/portfolio/summary` - Get portfolio summary
- `POST /api/v1/invest/base` - Execute base investment
- `POST /api/v1/invest/tactical` - Execute tactical investment

## Components

### Dashboard Components

1. **DashboardHeader** - Top navigation with user menu
2. **StatsCard** - Reusable stat display card
3. **TodayDecision** - Today's tactical decision display
4. **CapitalOverview** - Monthly capital breakdown
5. **PortfolioSummary** - Portfolio value and P&L
6. **SetCapitalModal** - Modal for setting capital
7. **BasePlanModal** - Modal for viewing base investment plan

### API Service

Centralized API service class:
```javascript
APIService.getCapital()
APIService.setCapital(amount)
APIService.generateBasePlan()
APIService.getTodayDecision()
APIService.getPortfolioSummary()
```

## Customization

### Colors

Edit `tailwind.config.js` to change color scheme:
```javascript
theme: {
  extend: {
    colors: {
      primary: {
        600: '#4f46e5', // Indigo
        // ... more shades
      }
    }
  }
}
```

### API URL

Change API base URL in `Dashboard.jsx`:
```javascript
const API_BASE_URL = 'https://your-api-domain.com';
```

## Features Implemented

### Dashboard View
- ✅ Quick stats (Total Invested, Current Value, Monthly Capital, Trading Days)
- ✅ Today's Decision card with decision type and NIFTY change
- ✅ Capital Overview with Base/Tactical split
- ✅ Portfolio Summary with P&L
- ✅ Quick action buttons

### Capital Management
- ✅ Set monthly capital modal
- ✅ Auto-calculate 60/40 split
- ✅ Trading days and daily tranche display
- ✅ View and update capital

### Base Plan
- ✅ Generate base investment plan
- ✅ ETF-wise breakdown with units
- ✅ Price information and allocation percentages
- ✅ Total investable amount calculation

### Responsive Design
- ✅ Mobile-first design
- ✅ Tablet and desktop layouts
- ✅ Collapsible mobile menu
- ✅ Touch-friendly interactions

## Development

### Hot Module Replacement
Vite provides instant HMR - changes appear immediately without refresh.

### Code Structure
- Keep components modular and reusable
- Use consistent naming conventions
- Follow React best practices
- Maintain proper error handling

### Adding New Features

1. Create new component in `src/`
2. Import and use in `Dashboard.jsx`
3. Add API method to `APIService` class
4. Test with backend running

## Deployment

### Docker Deployment (Recommended)

Create `Dockerfile`:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

### Static Hosting

Build files can be deployed to:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

## Troubleshooting

### CORS Issues
Ensure backend has CORS enabled for frontend origin.

### API Connection Failed
- Check backend is running on `http://localhost:8000`
- Verify Vite proxy configuration
- Check browser console for errors

### Build Errors
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

## License

Private - Part of ETF Assistant System

## Support

For issues or questions, check the main ETF Assistant documentation.
