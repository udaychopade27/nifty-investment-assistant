#!/bin/bash
# Quickstart Script for ETF Assistant - Unified Version
# Runs API + Scheduler + Telegram in one container

set -e  # Exit on error

echo "🚀 ETF Assistant Unified Quickstart"
echo "===================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your actual values (especially TELEGRAM_BOT_TOKEN)"
    echo ""
else
    echo "✅ .env file exists"
    echo ""
fi

# Build and start containers
echo "🐳 Building Docker containers..."
docker-compose build

echo ""
echo "🚀 Starting unified service (API + Scheduler + Telegram)..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 15

# Run database migrations
echo "📊 Running database migrations..."
docker-compose exec -T app alembic upgrade head 2>/dev/null || {
    echo "⚠️  Alembic migration had issues, creating tables directly..."
    docker-compose exec -T app python -c "
import asyncio
from app.infrastructure.db.database import init_db
asyncio.run(init_db())
print('✅ Tables created directly')
"
}

echo ""
echo "✅ System is ready!"
echo ""
echo "📍 Access points:"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - API Root: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - Services Status: http://localhost:8000/services/status"
echo ""
echo "🎯 Running Services:"
echo "   ✅ API Server (FastAPI)"
echo "   ✅ Scheduler (Daily decisions at 10 AM)"
echo "   ✅ Telegram Bot (if token configured)"
echo "   ✅ PostgreSQL Database"
echo ""
echo "📖 Next steps:"
echo "   1. Set your monthly capital: POST /api/v1/capital/set"
echo "   2. Configure Telegram token in .env (if not done)"
echo "   3. Send /start to your Telegram bot"
echo "   4. System will generate daily decisions at 10:00 AM"
echo ""
echo "🔧 Useful commands:"
echo "   - View logs: docker-compose logs -f app"
echo "   - Stop system: docker-compose down"
echo "   - Restart: docker-compose restart app"
echo "   - Shell access: docker-compose exec app bash"
echo ""
echo "📚 Read FINAL_FIXES.md for complete documentation"
echo ""echo "🚀 Enjoy using ETF Assistant!"