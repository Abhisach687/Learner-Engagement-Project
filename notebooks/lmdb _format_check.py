import lmdb
import os
import pickle
from pathlib import Path
import numpy as np
import binascii
import collections

def analyze_key_format(keys_sample):
    """Analyze a sample of keys to determine their most likely format"""
    format_counts = collections.Counter()
    key_examples = {}
    
    for key in keys_sample:
        # Check if key is likely UTF-8 string
        try:
            decoded = key.decode('utf-8')
            if decoded.startswith('video_'):
                format_counts['video_id_utf8'] += 1
                key_examples['video_id_utf8'] = key
            else:
                format_counts['other_utf8'] += 1
                key_examples['other_utf8'] = key
        except UnicodeDecodeError:
            pass
            
        # Check if key is pickled
        try:
            if key.startswith(b'\x80'):
                unpickled = pickle.loads(key)
                format_counts['pickle'] += 1
                key_examples['pickle'] = key
        except:
            pass
            
        # If not UTF-8 or pickle, count as binary
        if sum(format_counts.values()) == 0:
            format_counts['binary'] += 1
            key_examples['binary'] = key
    
    # Determine the most common format
    most_common_format = format_counts.most_common(1)[0][0] if format_counts else 'unknown'
    
    return {
        'most_common_format': most_common_format,
        'format_counts': dict(format_counts),
        'key_examples': key_examples
    }

# Path to cache directory
cache_dir = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project/cache")

# Find all LMDB directories
lmdb_dirs = [item for item in cache_dir.glob("lmdb_*") if item.is_dir() and (item / "data.mdb").exists()]

print(f"Found {len(lmdb_dirs)} LMDB databases in cache directory\n")

# Analyze each LMDB database
for lmdb_path in lmdb_dirs:
    print(f"=== Analyzing: {lmdb_path.name} ===")
    
    try:
        # Open the database
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
        
        # Get database stats
        print(f"Map size: {env.info()['map_size'] / (1024**3):.2f} GB")
        stat = env.stat()
        print(f"Entry count: {stat['entries']}")
        
        if stat['entries'] == 0:
            print("Database is empty.")
            env.close()
            continue
        
        # Sample keys (up to 50)
        keys_sample = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for i, (key, _) in enumerate(cursor):
                keys_sample.append(key)
                if i >= 49:  # Limit to 50 keys
                    break
        
        # Analyze key format
        analysis = analyze_key_format(keys_sample)
        print(f"Key format: {analysis['most_common_format']}")
        print(f"Format distribution: {analysis['format_counts']}")
        
        # Show example key
        if analysis['most_common_format'] in analysis['key_examples']:
            example_key = analysis['key_examples'][analysis['most_common_format']]
            print("Example key:")
            print(f"  Raw bytes: {binascii.hexlify(example_key)}")
            
            if analysis['most_common_format'] == 'video_id_utf8' or analysis['most_common_format'] == 'other_utf8':
                print(f"  Decoded: {example_key.decode('utf-8')}")
            elif analysis['most_common_format'] == 'pickle':
                try:
                    print(f"  Unpickled: {pickle.loads(example_key)}")
                except:
                    print("  Failed to unpickle key")
        
        # Sample first value
        with env.begin() as txn:
            cursor = txn.cursor()
            if cursor.first():
                key, value = cursor.item()
                print(f"First value size: {len(value)} bytes")
                try:
                    data = pickle.loads(value)
                    if isinstance(data, np.ndarray):
                        print(f"Value content: numpy array with shape {data.shape}, dtype {data.dtype}")
                    else:
                        print(f"Value content: {type(data)}")
                except:
                    print(f"Value content: Unable to unpickle, first bytes: {binascii.hexlify(value[:20])}")
        
        env.close()
        
    except Exception as e:
        print(f"Error analyzing database: {e}")
    
    print("\n")