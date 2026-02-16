#!/usr/bin/env python3
"""Reset all data for clean testing"""

from core.database import Database
import os

print('=== Resetting all data ===')

# 1. Clear database tables
db = Database()
with db.connection() as conn:
    # Clear all workflows
    result = conn.execute('DELETE FROM workflows')
    print(f'Cleared {result.rowcount} workflows')
    
    # Clear all templates (except system ones - keep those)
    result = conn.execute("DELETE FROM templates WHERE author != 'system'")
    print(f'Cleared {result.rowcount} user templates')
    
    # Clear comparisons table if it exists  
    try:
        result = conn.execute('DELETE FROM comparisons')
        print(f'Cleared {result.rowcount} comparisons')
    except Exception as e:
        print(f'Comparisons table not found: {e}')
    
    # Reset auto-increment counters
    conn.execute('DELETE FROM sqlite_sequence WHERE name IN ("workflows", "templates", "comparisons")')
    print('Reset auto-increment counters')

# 2. Clear coordinator memory
from src.deps import coordinator
coordinator.workflow_history.clear()
coordinator.workflow_stats = {
    'total_workflows': 0,
    'completed_workflows': 0,
    'error_workflows': 0,
    'success_rate': 0.0,
    'average_quality_score': 0.0,
    'average_processing_time': 0.0,
    'domain_distribution': {}
}
print('Cleared coordinator memory')

# 3. Clear cache if exists
try:
    from core.optimization import cache_manager
    cache_manager.clear()
    print('Cleared cache')
except Exception as e:
    print(f'Cache not available or already clear: {e}')

print('=== Data reset complete ===')
print('All user workflows, templates, and cached data have been cleared.')
print('System templates are preserved.')
print('You now have a clean state for testing!')