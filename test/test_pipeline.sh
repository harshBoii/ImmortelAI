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
        'https://www.linkedin.com/company/immortel-ai',
        'https://twitter.com/immortel_ai'
    ],
    'locale': 'en',
    'cluster_id': 'cluster-immortel-comparison',
    'published_at': None,
    'entity': {
        'name': 'Immortel AI',
        'oneLiner': 'Next-generation AI agents for enterprise automation',
        'website': 'https://example.com',
        'offerings': [
            'Autonomous document generation',
            'Intelligent workflow orchestration',
            'Multi-model AI pipelines'
        ],
        'differentiators': [
            'LangGraph-based stateful agents',
            'Human-in-the-loop review',
            'Schema.org native output'
        ],
        'competitors': ['Jasper AI', 'Writer.com', 'Copy.ai']
    },
    'intelligence': {
        'product_docs': (
            'Immortel AI uses LangGraph to build stateful, multi-step AI agents. '
            'It supports human-in-the-loop review via interrupt nodes. '
            'Pipelines produce schema.org-compliant FAQ and structured SEO pages. '
            'The platform handles 50k+ document generations per month for enterprise clients.'
        ),
        'market_research': (
            'The AI content generation market is projected to reach \$12B by 2027. '
            'Jasper AI focuses on marketing copy, Writer.com on brand consistency, '
            'and Copy.ai on sales enablement. Immortel AI differentiates with '
            'autonomous multi-step agents rather than single-prompt generation.'
        ),
        'customer_feedback': (
            'Enterprise users report 70% reduction in document turnaround time. '
            'The human-in-the-loop feature is cited as a key trust factor. '
            'Schema.org output directly improves search engine featured-snippet eligibility.'
        )
    },
    'query': 'immortel ai vs jasper ai for enterprise document automation',
    'page_type': 'COMPARISON',
    'existing_slugs': ['immortel-ai-overview'],
    'session_id': 'test-run-001'
}

print('▸ Invoking pipeline with COMPARISON page type...')
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
