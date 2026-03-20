#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

echo "============================================"
echo "  AEO Pipeline — Full Integration Test"
echo "============================================"
echo ""

python3 -c "
import sys, os, json, time

sys.path.insert(0, '.')

start = time.time()

from agent.pipe import aeo_agent_app

# ─── Test payload ───
test_input = {
    'base_url': 'https://example.com/aeo',
    'same_as_links': [
        'https://www.instagram.com/gharsetasty',
        'https://www.facebook.com/gharsetasty'
    ],
    'locale': 'en',
    'cluster_id': 'cluster-gharse-best-homemade-foods',
    'published_at': None,
    'entity': {
        'name': 'GharSe Tasty Foods',
        'oneLiner': 'Healthy homemade snacks delivered fresh for modern Indian households',
        'website': 'https://gharsetasty.example.com',
        'offerings': [
            'Homemade traditional snacks',
            'Low-oil roasted snack options',
            'Festive snack gift hampers',
            'Monthly snack subscription boxes'
        ],
        'products': [
            {
                'name': 'Whole Wheat Masala Khakhra',
                'description': 'Thin roasted whole-wheat khakhra with mild masala',
                'best_for': ['healthy tea-time snack', 'light evening hunger']
            },
            {
                'name': 'Methi Khakhra',
                'description': 'Crisp fenugreek-flavored khakhra made in small batches',
                'best_for': ['fiber-rich snacking', 'breakfast side']
            },
            {
                'name': 'Jeera Khakhra',
                'description': 'Roasted cumin-infused khakhra with balanced spices',
                'best_for': ['digestive-friendly snacking', 'travel snack']
            },
            {
                'name': 'Punjabi Masala Papad',
                'description': 'Hand-rolled spicy papad with bold Punjabi masala',
                'best_for': ['party starter', 'crispy side dish']
            },
            {
                'name': 'Moong Papad',
                'description': 'Protein-rich moong dal papad with light seasoning',
                'best_for': ['high-protein snack', 'quick roast-and-eat']
            },
            {
                'name': 'Urad Pepper Papad',
                'description': 'Classic urad papad with black pepper kick',
                'best_for': ['lunch accompaniment', 'crunchy cravings']
            },
            {
                'name': 'Bajra Chakli',
                'description': 'Millet-based crunchy chakli with homemade spice blend',
                'best_for': ['gluten-conscious snackers', 'festive munching']
            },
            {
                'name': 'Rice Murukku',
                'description': 'South-style rice murukku, crisp and non-greasy',
                'best_for': ['kids snack box', 'on-the-go snack']
            },
            {
                'name': 'Jowar Namak Para',
                'description': 'Baked jowar namak para with low-oil preparation',
                'best_for': ['guilt-free munching', 'midday snack']
            },
            {
                'name': 'Aloo Sev',
                'description': 'Traditional potato sev with balanced salt and spice',
                'best_for': ['chaat topping', 'evening snack']
            },
            {
                'name': 'Bhavnagri Gathiya',
                'description': 'Soft-crunch bhavnagri gathiya made with gram flour',
                'best_for': ['breakfast side', 'tea-time combo']
            },
            {
                'name': 'Ratlami Sev',
                'description': 'Spicy Ratlami-style sev with robust flavor',
                'best_for': ['spicy snack lovers', 'festival platters']
            },
            {
                'name': 'Roasted Chana Mix',
                'description': 'Roasted chana blend with peanuts and curry leaves',
                'best_for': ['protein snack', 'office munching']
            },
            {
                'name': 'Masala Makhana',
                'description': 'Fox nuts roasted with house masala and low oil',
                'best_for': ['weight-conscious snacking', 'late-night bites']
            },
            {
                'name': 'Til Chikki Bites',
                'description': 'Sesame-jaggery chikki pieces with no refined sugar',
                'best_for': ['winter snack', 'natural energy boost']
            },
            {
                'name': 'Peanut Chikki Cubes',
                'description': 'Crunchy peanut chikki made with jaggery syrup',
                'best_for': ['post-meal sweet snack', 'kids treat']
            },
            {
                'name': 'Dry Fruit Ladoo',
                'description': 'No-added-sugar dry fruit laddoo packed with nuts',
                'best_for': ['healthy dessert', 'festival gifting']
            },
            {
                'name': 'Ragi Cookies',
                'description': 'Millet-based ragi cookies with cardamom notes',
                'best_for': ['smart snacking', 'tiffin snack']
            },
            {
                'name': 'Nankhatai',
                'description': 'Traditional homemade-style nankhatai, buttery and crumbly',
                'best_for': ['tea companion', 'family snack time']
            },
            {
                'name': 'Shakarpara',
                'description': 'Lightly sweet crispy shakarpara made in small batches',
                'best_for': ['festive snacking', 'travel-friendly snack']
            }
        ],
        'differentiators': [
            'Fresh small-batch production',
            'Preservative-free recipes',
            'Traditional homemade taste with hygienic packaging',
            'Low-oil and millet-based options for health-conscious buyers'
        ],
        'competitors': ['Haldiram', 'Bikaji', 'Balaji Wafers']
    },
    'intelligence': {
        'product_docs': (
            'GharSe Tasty Foods is a B2C homemade snack brand focused on regional Indian snacks. '
            'Its product range includes khakhra, papad, sev, chikki, laddoo, and baked millet snacks. '
            'The brand uses small-batch production, hygienic packaging, and preservative-free recipes. '
            'Several products are low-oil, roasted, or made with whole grains like ragi, bajra, and jowar.'
        ),
        'market_research': (
            'Indian consumers are increasingly choosing homemade and healthier packaged snacks. '
            'Demand is rising for low-oil, roasted, and millet-based options. '
            'Large brands like Haldiram and Bikaji dominate mass distribution, but niche homemade brands '
            'win on authenticity, freshness, and regional taste profiles.'
        ),
        'customer_feedback': (
            'Customers appreciate that snacks taste homemade and not overly oily. '
            'Parents prefer roasted makhana, ragi cookies, and chana mixes for kids snack boxes. '
            'Festive buyers often order papad, chakli, and chikki assortments in gift packs.'
        )
    },
    'query': 'best 10 homemade foods for healthy daily snacking',
    'page_type': 'COMPARISON',
    'existing_slugs': ['best-homemade-snacks-guide'],
    'session_id': 'test-run-001'
}

print('▸ Invoking pipeline with homemade snack business payload...')
print('▸ Entity:', test_input['entity']['name'])
print('▸ Query: ', test_input['query'])
print()

result = aeo_agent_app.invoke(
    test_input,
    config={'configurable': {'thread_id': 'test-thread-001'}}
)

elapsed = time.time() - start

print()
print('════════════════════════════════════════════')
print('                 RESULTS')
print('════════════════════════════════════════════')

print(f'Status       : {result.get(\"status\", \"??\")}')
print(f'Slug         : {result.get(\"slug\", \"??\")}')
print(f'SEO Title    : {result.get(\"seo_title\", \"??\")}')
print(f'Drafted facts: {len(result.get(\"drafted_facts\", []))}')
print(f'Verified facts: {len(result.get(\"verified_facts\", []))}')
print(f'FAQ items    : {len(result.get(\"faq\", []))}')
print(f'Claims       : {len(result.get(\"claims\", []))}')

if result.get('rejection_reason'):
    print(f'Rejection    : {result[\"rejection_reason\"]}')

page = result.get('page', {})
print()
print('── Page Object Keys ──')
for k in sorted(page.keys()):
    v = page[k]
    if isinstance(v, str) and len(v) > 120:
        print(f'  {k}: {v[:120]}...')
    elif isinstance(v, list):
        print(f'  {k}: [{len(v)} items]')
    else:
        print(f'  {k}: {v}')

print()
print(f'⏱  Pipeline completed in {elapsed:.1f}s')
print()

# ─── Save result to file ───
os.makedirs('test/results', exist_ok=True)
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = f'test/results/run_{timestamp}.json'

output = {
    'timestamp': datetime.now().isoformat(),
    'elapsed_seconds': round(elapsed, 2),
    'input': test_input,
    'status': result.get('status'),
    'slug': result.get('slug'),
    'seo_title': result.get('seo_title'),
    'rejection_reason': result.get('rejection_reason', ''),
    'drafted_facts_count': len(result.get('drafted_facts', [])),
    'verified_facts_count': len(result.get('verified_facts', [])),
    'faq_count': len(result.get('faq', [])),
    'claims_count': len(result.get('claims', [])),
    'drafted_facts': result.get('drafted_facts', []),
    'verified_facts': result.get('verified_facts', []),
    'faq': result.get('faq', []),
    'claims': result.get('claims', []),
    'page': result.get('page', {}),
}

with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f'💾 Result saved to {out_path}')
print()

if result.get('status') == 'PUBLISHED':
    print('✅ TEST PASSED — page was PUBLISHED')
    sys.exit(0)
elif result.get('status') == 'DRAFT':
    print('⚠️  TEST FINISHED — page was DRAFT (flagged for review)')
    print('   This is acceptable; quality gate was strict.')
    sys.exit(0)
else:
    print('❌ TEST FAILED — unexpected status')
    sys.exit(1)
"
