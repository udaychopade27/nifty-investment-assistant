#!/bin/bash
# Quickstart Script for ETF Assistant
# Run this to get the system up and running quickly

set -e  # Exit on error

echo "🚀 ETF Assistant Quickstart"
echo "================================"
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
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for database to be ready..."
sleep 10

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
echo "   - pgAdmin (optional): http://localhost:5050"
echo ""
echo "📖 Next steps:"
echo "   1. Set your monthly capital: Use Telegram bot or API"
echo "   2. System will generate daily decisions at 10:00 AM"
echo "   3. Review decisions and execute trades manually"
echo ""
echo "🔧 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop system: docker-compose down"
echo "   - Restart: docker-compose restart"
echo "   - Shell access: docker-compose exec app bash"
echo ""
echo "📚 Read README.md for complete documentation"
echo ""
