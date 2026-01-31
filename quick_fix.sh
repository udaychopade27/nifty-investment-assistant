#!/bin/bash
# Quick fix for AssetClass issue
# Run this if you get: 'bond' is not a valid AssetClass

echo "🔧 Applying quick fix for AssetClass..."

# Fix the BHARATBOND asset class
sed -i "s/asset_class: bond/asset_class: debt/g" config/etfs.yml

echo "✅ Fix applied!"
echo ""
echo "Now run: python test_system.py"
