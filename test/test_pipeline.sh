#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

CONTENT_TYPE="${1:-BLOG}"

echo "============================================"
echo "  AEO Pipeline — Full Integration Test"
echo "  content_type: ${CONTENT_TYPE}"
echo "============================================"
echo ""

python3 -c "
import sys, os, json, time

sys.path.insert(0, '.')

start = time.time()

from agent.pipe import aeo_agent_app
from agent.functions import preflight

CONTENT_TYPE = os.environ.get('CONTENT_TYPE_OVERRIDE', '${CONTENT_TYPE}')

# ─── Shared test payload base ───
test_input = {
    'base_url': 'https://example.com/aeo',
    'same_as_links': [
        'https://www.instagram.com/gharsetasty',
        'https://www.facebook.com/gharsetasty'
    ],
    'locale': 'en',
    'cluster_id': 'cluster-gharse-best-homemade-foods',
    'published_at': None,
    'content_type': CONTENT_TYPE,
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
                'name': 'Masala Makhana',
                'description': 'Fox nuts roasted with house masala and low oil',
                'best_for': ['weight-conscious snacking', 'late-night bites']
            },
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
    'query': 'best homemade foods for healthy daily snacking',
    'page_type': 'COMPARISON',
    'existing_slugs': ['best-homemade-snacks-guide'],
    'session_id': f'test-run-{CONTENT_TYPE.lower()}'
}

print(f'▸ Invoking pipeline with content_type={CONTENT_TYPE}')
print('▸ Entity:', test_input['entity']['name'])
print('▸ Query: ', test_input['query'])
print()

state_in = preflight(test_input)
result = aeo_agent_app.invoke(
    state_in,
    config={'configurable': {'thread_id': f'test-thread-{CONTENT_TYPE}'}}
)

elapsed = time.time() - start

print()
print('════════════════════════════════════════════')
print('                 RESULTS')
print('════════════════════════════════════════════')

print(f'Content Type : {result.get(\"content_type\", CONTENT_TYPE)}')
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
print('── Page / Content Object Keys ──')
for k in sorted(page.keys()):
    v = page[k]
    if isinstance(v, str) and len(v) > 120:
        print(f'  {k}: {v[:120]}...')
    elif isinstance(v, list):
        print(f'  {k}: [{len(v)} items]')
    else:
        print(f'  {k}: {v}')

if page.get('text'):
    print()
    print('── Post Text ──')
    print(page['text'])
    print(f'Char count: {page.get(\"charCount\", len(page[\"text\"]))}')

print()
print(f'⏱  Pipeline completed in {elapsed:.1f}s')
print()

# ─── Save result to file ───
os.makedirs('test/results', exist_ok=True)
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = f'test/results/run_{CONTENT_TYPE.lower()}_{timestamp}.json'

output = {
    'timestamp': datetime.now().isoformat(),
    'elapsed_seconds': round(elapsed, 2),
    'content_type': result.get('content_type', CONTENT_TYPE),
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
    print('✅ TEST PASSED — content was PUBLISHED')
    sys.exit(0)
elif result.get('status') == 'DRAFT':
    print('⚠️  TEST FINISHED — content was DRAFT (flagged for review)')
    print('   This is acceptable; quality gate was strict.')
    sys.exit(0)
else:
    print('❌ TEST FAILED — unexpected status')
    sys.exit(1)
"
