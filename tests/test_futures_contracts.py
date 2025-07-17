"""
Tests for futures contracts specifications
"""

import pytest
from futures_contracts import FuturesContract, get_contract, get_all_contracts

class TestFuturesContracts:
    
    def test_futures_contract_creation(self):
        """Test creating a futures contract"""
        contract = FuturesContract(
            symbol="TEST",
            name="Test Future",
            exchange="TEST_EXCHANGE",
            tick_size=0.25,
            tick_value=12.50,
            margin_requirement=5000,
            trading_hours="24/5",
            contract_months="HMUZ",
            description="Test contract"
        )
        
        assert contract.symbol == "TEST"
        assert contract.name == "Test Future"
        assert contract.tick_size == 0.25
        assert contract.tick_value == 12.50
        assert contract.point_value == 50.0  # tick_value / tick_size
    
    def test_get_nq_contract(self):
        """Test getting NQ futures contract"""
        contract = get_contract("NQ")
        
        assert contract is not None
        assert contract.symbol == "NQ"
        assert contract.name == "E-mini Nasdaq-100"
        assert contract.exchange == "CME"
        assert contract.tick_size == 0.25
        assert contract.tick_value == 5.0
        assert contract.margin_requirement == 17600
    
    def test_get_es_contract(self):
        """Test getting ES futures contract"""
        contract = get_contract("ES")
        
        assert contract is not None
        assert contract.symbol == "ES"
        assert contract.name == "E-mini S&P 500"
        assert contract.exchange == "CME"
        assert contract.tick_size == 0.25
        assert contract.tick_value == 12.50
        assert contract.margin_requirement == 13200
    
    def test_get_all_contracts(self):
        """Test getting all available contracts"""
        contracts = get_all_contracts()
        
        assert isinstance(contracts, dict)
        assert len(contracts) >= 2  # At least NQ and ES
        assert "NQ" in contracts
        assert "ES" in contracts
        
        # Check all contracts have required fields
        for symbol, contract in contracts.items():
            assert hasattr(contract, 'symbol')
            assert hasattr(contract, 'tick_size')
            assert hasattr(contract, 'tick_value')
            assert hasattr(contract, 'margin_requirement')
    
    def test_get_nonexistent_contract(self):
        """Test getting a non-existent contract"""
        contract = get_contract("INVALID")
        assert contract is None
    
    def test_contract_calculations(self):
        """Test contract value calculations"""
        nq = get_contract("NQ")
        
        # Test point value calculation
        assert nq.point_value == 20.0  # $5 per 0.25 tick = $20 per point
        
        # Test value calculations
        # 10 tick movement
        ticks = 10
        value = ticks * nq.tick_value
        assert value == 50.0  # 10 * $5
        
        # 2.5 point movement (10 ticks)
        points = 2.5
        value = points * nq.point_value
        assert value == 50.0  # 2.5 * $20
    
    def test_contract_string_representation(self):
        """Test contract string representation"""
        contract = get_contract("NQ")
        str_repr = str(contract)
        
        assert "NQ" in str_repr
        assert "E-mini Nasdaq-100" in str_repr
    
    def test_all_contract_specs_valid(self):
        """Test that all contract specifications are valid"""
        contracts = get_all_contracts()
        
        for symbol, contract in contracts.items():
            # All numeric values should be positive
            assert contract.tick_size > 0
            assert contract.tick_value > 0
            assert contract.margin_requirement > 0
            assert contract.point_value > 0
            
            # Strings should not be empty
            assert len(contract.symbol) > 0
            assert len(contract.name) > 0
            assert len(contract.exchange) > 0
            assert len(contract.trading_hours) > 0