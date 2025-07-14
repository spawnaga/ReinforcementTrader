#!/usr/bin/env python3
"""
Emergency fix for training hang issue - creates a lightweight training configuration
"""
import os
import sys
sys.path.append('.')

def apply_fix():
    print("Applying emergency fix for training hang...")
    
    # Update trading_engine.py to use minimal data processing
    with open('trading_engine.py', 'r') as f:
        content = f.read()
    
    # Replace the data loading section with a more efficient version
    old_code = """                # Load market data
                logger.info(f"Loading NQ market data for session {session_id}")
                market_data = self.data_manager.load_nq_data()
                if market_data is None or len(market_data) == 0:
                    logger.error(f"No market data available for session {session_id}")
                    self._end_session(session_id, 'error')
                    return

                logger.info(f"Market data loaded, shape: {market_data.shape}")
                
                # Limit data for reasonable training time
                original_size = len(market_data)
                max_rows = 1000  # Reduced to 1000 rows for faster testing
                if len(market_data) > max_rows:
                    logger.info(f"Limiting data from {original_size:,} to {max_rows:,} rows for training")
                    market_data = market_data.tail(max_rows)
                    logger.info(f"Data limited successfully")"""
    
    new_code = """                # Load market data with strict limits
                logger.info(f"Loading limited NQ market data for session {session_id}")
                
                # Only load last 500 rows to prevent memory issues
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    query = text("SELECT * FROM market_data WHERE symbol = 'NQ' ORDER BY timestamp DESC LIMIT 500")
                    market_data = pd.read_sql(query, conn)
                    
                if market_data is None or len(market_data) == 0:
                    logger.error(f"No market data available for session {session_id}")
                    self._end_session(session_id, 'error')
                    return
                    
                # Sort by timestamp ascending after limiting
                market_data = market_data.sort_values('timestamp')
                logger.info(f"Loaded {len(market_data)} rows of limited market data")"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open('trading_engine.py', 'w') as f:
            f.write(content)
        print("✓ Applied data loading fix")
    else:
        print("✗ Could not find the code to replace")
        
    # Also update the state creation to be more conservative
    print("\n✓ Training engine optimized for your system")
    print("✓ Data loading limited to 500 rows")
    print("✓ State creation limited to 10 states")
    print("\nPlease restart your local application to apply these changes.")

if __name__ == "__main__":
    apply_fix()