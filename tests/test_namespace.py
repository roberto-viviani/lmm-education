"""Test script to verify namespace flattening works with Pylance."""

import lmm_education as lme

# Test that all exported functions are accessible
print("Testing namespace flattening...")

# Configuration
assert hasattr(lme, 'create_default_config_file')

# Content processing
assert hasattr(lme, 'scan')
assert hasattr(lme, 'scan_messages')
assert hasattr(lme, 'scan_clear_messages')
assert hasattr(lme, 'scan_rag')

# Database operations
assert hasattr(lme, 'ingest')
assert hasattr(lme, 'querydb')
assert hasattr(lme, 'query')
assert hasattr(lme, 'database_info')

# Test __all__ is properly defined
assert hasattr(lme, '__all__')
assert len(lme.__all__) == 10

# Verify internal modules are still accessible but not in __all__
assert 'config' in dir(lme)  # Internal module
assert 'config' not in lme.__all__  # But not in public API

print("✓ All public API functions accessible")
print(f"✓ Public API has {len(lme.__all__)} functions")
print(f"✓ Version: {lme.__version__}")
print("\nNamespace flattening working correctly!")
