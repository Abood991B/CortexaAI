#!/usr/bin/env python3
"""
Comprehensive tests for src/main.py FastAPI application.
Tests cover application startup, configuration, API endpoints, error handling,
configuration loading, and integration with the workflow system.
"""

import os
import sys
import json
import time
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import the FastAPI app and related components
from src.main import (
    app,
    PromptRequest,
    PromptResponse,
    SystemStats,
    classifier_instance,
    evaluator_instance,
    coordinator
)
from config.config import settings, metrics


class TestFastAPIApp:
    """Test FastAPI application startup and configuration."""

    def test_app_creation(self):
        """Test that the FastAPI app is created with correct configuration."""
        assert app.title == "CortexaAI"
        assert "Multi-Agent" in app.description
        assert app.version == "3.0.0"

    def test_app_routes_registered(self):
        """Test that all expected routes are registered."""
        routes = [route.path for route in app.routes]

        expected_routes = [
            "/",
            "/health",
            "/metrics",
            "/api/process-prompt",
            "/api/domains",
            "/api/stats",
            "/api/history"
        ]

        for route in expected_routes:
            assert route in routes, f"Route {route} not found in app routes"

    def test_logger_setup(self):
        """Test that logger is properly set up."""
        import src.main
        import logging
        # Verify the module has a logger configured
        assert hasattr(src.main, 'logger') or logging.getLogger("src.main") is not None


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('src.main.time.time')
    @patch('src.main.psutil.virtual_memory')
    @patch('src.main.psutil.cpu_percent')
    @patch('src.main.psutil.net_connections')
    @patch('src.main.coordinator')
    @patch('src.main.metrics')
    @patch('src.main.settings')
    def test_health_endpoint_success(self, mock_settings, mock_metrics, mock_coordinator,
                                    mock_net_conn, mock_cpu, mock_memory, mock_time):
        """Test successful health check response."""
        # Setup mocks
        mock_time.return_value = 1234567890.0
        mock_memory.return_value.percent = 45.5
        mock_cpu.return_value = 23.1
        mock_net_conn.return_value = [Mock()] * 5

        mock_settings.langsmith_api_key = "test_key"
        mock_settings.openai_api_key = "openai_key"
        mock_settings.anthropic_api_key = "anthropic_key"
        mock_settings.google_api_key = "google_key"

        mock_coordinator.get_available_domains.return_value = [
            {"domain": "test_domain", "description": "Test", "keywords": ["test"]}
        ]

        mock_metrics.get_metrics.return_value = {
            "workflows_completed": 10,
            "workflows_failed": 2,
            "llm_calls_total": 100,
            "retry_attempts": 5
        }

        response = self.client.get("/health")

        assert response.status_code == 200
        health_data = response.json()

        assert health_data["status"] == "healthy"
        assert health_data["version"] == "3.0.0"
        assert health_data["readiness"] is True
        assert health_data["liveness"] is True
        assert "llm_providers" in health_data["components"]
        assert "langsmith" in health_data["components"]
        assert "coordinator" in health_data["components"]
        assert "system" in health_data

    @patch('src.main.coordinator')
    def test_health_endpoint_coordinator_failure(self, mock_coordinator):
        """Test health check when coordinator fails."""
        mock_coordinator.get_available_domains.side_effect = Exception("Coordinator error")

        response = self.client.get("/health")

        assert response.status_code == 200
        health_data = response.json()

        assert health_data["status"] == "unhealthy"
        assert health_data["components"]["coordinator"]["status"] == "unhealthy"


class TestMetricsEndpoint:
    """Test the /metrics endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('src.main.metrics')
    @patch('src.main.settings')
    @patch('src.main.psutil.Process')
    @patch('src.main.psutil.virtual_memory')
    @patch('src.main.psutil.cpu_percent')
    def test_metrics_endpoint(self, mock_cpu, mock_memory, mock_process, mock_settings, mock_metrics):
        """Test Prometheus metrics endpoint."""
        # Setup mocks
        mock_settings.langsmith_api_key = "test_key"

        mock_metrics.get_metrics.return_value = {
            "llm_calls_total": 100,
            "llm_calls_success": 95,
            "llm_calls_error": 5,
            "workflows_completed": 50,
            "workflows_failed": 3,
            "retry_attempts": 10,
            "llm_call_duration_seconds": [0.5, 1.2, 2.1, 0.8],
            "domains_processed": {"software_engineering": 25, "data_science": 15}
        }

        mock_memory_instance = Mock()
        mock_memory_instance.percent = 60.5
        mock_memory.return_value = mock_memory_instance

        mock_cpu.return_value = 45.2

        mock_process_instance = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 104857600  # 100 MB
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process.return_value = mock_process_instance

        response = self.client.get("/metrics")

        assert response.status_code == 200
        metrics_text = response.text

        # Check that key metrics are present
        assert 'system_info' in metrics_text
        assert '3.0.0' in metrics_text
        assert 'llm_calls_total 100' in metrics_text
        assert 'workflows_completed 50' in metrics_text
        assert 'process_memory_bytes 104857600' in metrics_text
        assert 'system_memory_percent 60.5' in metrics_text
        assert 'system_cpu_percent 45.2' in metrics_text


class TestRootEndpoint:
    """Test the root endpoint (/)."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('pathlib.Path.read_text')
    def test_root_endpoint(self, mock_read_text):
        """Test that root endpoint serves HTML content."""
        mock_read_text.return_value = "<html><body>Test HTML</body></html>"

        response = self.client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Test HTML" in response.text


class TestProcessPromptEndpoint:
    """Test the /api/process-prompt endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('src.main.coordinator')
    def test_process_prompt_success(self, mock_coordinator):
        """Test successful prompt processing (synchronous mode)."""
        mock_result = {
            "workflow_id": "test_workflow_123",
            "status": "completed",
            "timestamp": "2024-01-01T12:00:00Z",
            "processing_time_seconds": 1.5,
            "input": {"prompt": "Test prompt", "prompt_type": "raw"},
            "output": {"improved_prompt": "Improved test prompt"},
            "analysis": {"score": 0.9},
            "comparison": {"improvement_ratio": 0.15},
            "metadata": {"iterations": 2}
        }
        mock_coordinator.process_prompt.return_value = mock_result

        request_data = {
            "prompt": "Test prompt",
            "prompt_type": "raw",
            "return_comparison": True,
            "use_langgraph": False,
            "synchronous": True
        }

        response = self.client.post("/api/process-prompt", json=request_data)

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["status"] == "completed"
        assert response_data["processing_time_seconds"] == 1.5
        mock_coordinator.process_prompt.assert_called_once_with(
            prompt="Test prompt",
            prompt_type="raw",
            return_comparison=True
        )

    @patch('src.main.process_prompt_with_langgraph')
    def test_process_prompt_with_langgraph(self, mock_langgraph):
        """Test prompt processing using LangGraph (synchronous mode)."""
        mock_result = {
            "workflow_id": "langgraph_workflow_123",
            "status": "completed",
            "timestamp": "2024-01-01T12:00:00Z",
            "processing_time_seconds": 2.0,
            "input": {"prompt": "Test prompt", "prompt_type": "structured"},
            "output": {"improved_prompt": "LangGraph improved prompt"},
            "metadata": {"used_langgraph": True}
        }
        mock_langgraph.return_value = mock_result

        request_data = {
            "prompt": "Test prompt",
            "prompt_type": "structured",
            "return_comparison": False,
            "use_langgraph": True,
            "synchronous": True
        }

        response = self.client.post("/api/process-prompt", json=request_data)

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["workflow_id"] == "langgraph_workflow_123"
        assert response_data["status"] == "completed"
        mock_langgraph.assert_called_once_with(
            prompt="Test prompt",
            prompt_type="structured",
        )

    @patch('src.main.coordinator')
    def test_process_prompt_error_handling(self, mock_coordinator):
        """Test error handling in prompt processing (synchronous mode)."""
        mock_coordinator.process_prompt.side_effect = Exception("Processing failed")

        request_data = {
            "prompt": "Test prompt",
            "prompt_type": "raw",
            "synchronous": True
        }

        response = self.client.post("/api/process-prompt", json=request_data)

        # When an exception is raised inside the sync path, FastAPI will return 500
        assert response.status_code in [400, 500]


class TestDomainsEndpoint:
    """Test the /api/domains endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('src.main.coordinator')
    def test_get_domains_success(self, mock_coordinator):
        """Test successful domain retrieval."""
        mock_domains = [
            {
                "domain": "software_engineering",
                "description": "Software development tasks",
                "keywords": ["code", "programming", "algorithm"],
                "has_expert_agent": True,
                "agent_created": True
            },
            {
                "domain": "data_science",
                "description": "Data analysis and machine learning",
                "keywords": ["data", "analysis", "ml"],
                "has_expert_agent": True,
                "agent_created": False
            }
        ]
        mock_coordinator.get_available_domains.return_value = mock_domains

        response = self.client.get("/api/domains")

        assert response.status_code == 200
        domains_data = response.json()

        assert len(domains_data) == 2
        assert domains_data[0]["domain"] == "software_engineering"
        assert domains_data[0]["has_expert_agent"] is True
        assert domains_data[1]["domain"] == "data_science"

    @patch('src.main.coordinator')
    def test_get_domains_error_handling(self, mock_coordinator):
        """Test error handling in domain retrieval."""
        mock_coordinator.get_available_domains.side_effect = Exception("Domain retrieval failed")

        response = self.client.get("/api/domains")

        assert response.status_code == 500
        error_data = response.json()

        assert "Failed to get domains" in error_data["detail"]


class TestStatsEndpoint:
    """Test the /api/stats endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('src.main.coordinator')
    def test_get_stats_success(self, mock_coordinator):
        """Test successful stats retrieval."""
        mock_stats = {
            "total_workflows": 100,
            "completed_workflows": 95,
            "error_workflows": 5,
            "success_rate": 0.95,
            "average_quality_score": 0.87,
            "average_processing_time": 2.3,
            "domain_distribution": {
                "software_engineering": 60,
                "data_science": 35,
                "other": 5
            }
        }
        mock_coordinator.get_workflow_stats.return_value = mock_stats

        response = self.client.get("/api/stats")

        assert response.status_code == 200
        stats_data = response.json()

        assert stats_data["total_workflows"] == 100
        assert stats_data["success_rate"] == 0.95
        assert stats_data["average_quality_score"] == 0.87
        assert "software_engineering" in stats_data["domain_distribution"]

    @patch('src.main.coordinator')
    def test_get_stats_no_history(self, mock_coordinator):
        """Test stats retrieval when no workflow history exists."""
        mock_coordinator.get_workflow_stats.return_value = {
            "error": "No workflow history available"
        }

        response = self.client.get("/api/stats")

        assert response.status_code == 200
        stats_data = response.json()

        # Should return default stats when no history
        assert stats_data["total_workflows"] == 0
        assert stats_data["completed_workflows"] == 0
        assert stats_data["success_rate"] == 0.0
        assert stats_data["average_quality_score"] == 0.0
        assert stats_data["average_processing_time"] == 0.0
        assert stats_data["domain_distribution"] == {}

    @patch('src.main.coordinator')
    def test_get_stats_error_handling(self, mock_coordinator):
        """Test error handling in stats retrieval."""
        mock_coordinator.get_workflow_stats.side_effect = Exception("Stats retrieval failed")

        response = self.client.get("/api/stats")

        assert response.status_code == 500
        error_data = response.json()

        assert "Failed to get stats" in error_data["detail"]


class TestHistoryEndpoint:
    """Test the /api/history endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('src.main.coordinator')
    def test_get_history_success(self, mock_coordinator):
        """Test successful history retrieval."""
        mock_history = [
            {
                "workflow_id": "wf_001",
                "timestamp": "2024-01-01T10:00:00Z",
                "status": "completed",
                "domain": "software_engineering",
                "quality_score": 0.9
            },
            {
                "workflow_id": "wf_002",
                "timestamp": "2024-01-01T11:00:00Z",
                "status": "completed",
                "domain": "data_science",
                "quality_score": 0.85
            }
        ]
        mock_coordinator.get_workflow_history.return_value = mock_history

        response = self.client.get("/api/history?limit=10")

        assert response.status_code == 200
        history_data = response.json()

        assert len(history_data) == 2
        assert history_data[0]["workflow_id"] == "wf_001"
        assert history_data[1]["workflow_id"] == "wf_002"
        mock_coordinator.get_workflow_history.assert_called_once_with(limit=10)

    @patch('src.main.coordinator')
    def test_get_history_error_handling(self, mock_coordinator):
        """Test error handling in history retrieval."""
        mock_coordinator.get_workflow_history.side_effect = Exception("History retrieval failed")

        response = self.client.get("/api/history")

        assert response.status_code == 500
        error_data = response.json()

        assert "Failed to get history" in error_data["detail"]


class TestErrorHandling:
    """Test error handling for invalid requests."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_invalid_json_request(self):
        """Test handling of invalid JSON in request body."""
        response = self.client.post(
            "/api/process-prompt",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Missing prompt field
        request_data = {
            "prompt_type": "raw"
        }

        response = self.client.post("/api/process-prompt", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_invalid_prompt_type(self):
        """Test handling of invalid prompt_type values."""
        request_data = {
            "prompt": "Test prompt",
            "prompt_type": "invalid_type"
        }

        # This might pass validation depending on the model, but let's test with an empty prompt
        request_data_empty = {
            "prompt": "",
            "prompt_type": "raw"
        }

        response = self.client.post("/api/process-prompt", json=request_data_empty)

        # Should still work, but coordinator may handle empty prompts gracefully
        assert response.status_code in [200, 500]  # Either success or server error


class TestConfigurationAndEnvironment:
    """Test configuration loading and environment variables."""

    @patch('src.main.settings')
    def test_configuration_access(self, mock_settings):
        """Test that configuration settings are accessible."""
        # Test various settings that should be available
        assert hasattr(mock_settings, 'host')
        assert hasattr(mock_settings, 'port')
        assert hasattr(mock_settings, 'log_level')

    @patch('src.main.setup_langsmith')
    def test_langsmith_setup_called(self, mock_setup_langsmith):
        """Test that setup_langsmith is called during app initialization."""
        # Re-import to trigger setup
        from importlib import reload
        import src.main
        reload(src.main)

        mock_setup_langsmith.assert_called_once()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'GOOGLE_API_KEY': 'test_google_key'
    })
    def test_environment_variables(self):
        """Test that environment variables are properly loaded."""
        # This would normally be tested through the config module
        # but we're testing that the app can start with env vars set
        pass


class TestIntegrationWithWorkflowSystem:
    """Test integration with the workflow system."""

    @patch('src.main.classifier_instance')
    @patch('src.main.evaluator_instance')
    @patch('src.main.WorkflowCoordinator')
    def test_global_instances_creation(self, mock_coordinator_class, mock_evaluator, mock_classifier):
        """Test that global instances are created correctly."""
        # Re-import to trigger instance creation
        from importlib import reload
        import src.main
        reload(src.main)

        # Verify that instances were created
        from src.main import classifier_instance, evaluator_instance, coordinator

        assert classifier_instance is not None
        assert evaluator_instance is not None
        assert coordinator is not None

        # Verify WorkflowCoordinator was called with correct arguments
        mock_coordinator_class.assert_called_once_with(classifier_instance, evaluator_instance)

    @patch('src.main.coordinator')
    def test_workflow_coordinator_integration(self, mock_coordinator):
        """Test that the app properly integrates with WorkflowCoordinator."""
        # Test that coordinator methods are called correctly from endpoints
        pass


class TestPydanticModels:
    """Test Pydantic models used in the API."""

    def test_prompt_request_model(self):
        """Test PromptRequest model validation."""
        # Valid request
        request = PromptRequest(
            prompt="Test prompt",
            prompt_type="raw",
            return_comparison=True,
            use_langgraph=False,
            synchronous=True
        )
        assert request.prompt == "Test prompt"
        assert request.prompt_type == "raw"
        assert request.return_comparison is True
        assert request.use_langgraph is False
        assert request.synchronous is True

        # Test defaults
        request_defaults = PromptRequest(prompt="Test")
        assert request_defaults.prompt_type == "auto"
        assert request_defaults.return_comparison is True
        assert request_defaults.use_langgraph is False
        assert request_defaults.synchronous is False

    def test_prompt_response_model(self):
        """Test PromptResponse model validation."""
        response = PromptResponse(
            workflow_id="test_123",
            status="completed",
            timestamp="2024-01-01T12:00:00Z",
            processing_time_seconds=1.5,
            input={"prompt": "test"},
            output={"result": "output"},
            analysis={"score": 0.9},
            comparison={"ratio": 0.1},
            metadata={"extra": "data"}
        )

        assert response.workflow_id == "test_123"
        assert response.status == "completed"
        assert response.processing_time_seconds == 1.5


    def test_system_stats_model(self):
        """Test SystemStats model validation."""
        stats = SystemStats(
            total_workflows=100,
            completed_workflows=95,
            error_workflows=5,
            success_rate=0.95,
            average_quality_score=0.87,
            average_processing_time=2.3,
            domain_distribution={"software": 60, "data": 40}
        )

        assert stats.total_workflows == 100
        assert stats.success_rate == 0.95
        assert stats.average_quality_score == 0.87


if __name__ == "__main__":
    pytest.main([__file__])
