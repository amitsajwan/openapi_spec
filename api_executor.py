# api_executor.py
import httpx
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class APIExecutor:
    """
    Handles the actual execution of API calls based on node definitions.
    Uses httpx for asynchronous HTTP requests.
    """
    def __init__(self, base_url: Optional[str] = None, default_headers: Optional[Dict[str, str]] = None, timeout: float = 30.0):
        """
        Initializes the APIExecutor.

        Args:
            base_url (Optional[str]): A base URL to prepend to endpoints if they are relative.
                                      If None, endpoints are expected to be absolute URLs.
            default_headers (Optional[Dict[str, str]]): Default headers to include in every request.
            timeout (float): Default timeout for HTTP requests in seconds.
        """
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.timeout = timeout
        # Initialize an async HTTP client. It's good practice to reuse the client.
        self._client = httpx.AsyncClient(timeout=self.timeout)
        logger.info(f"APIExecutor initialized. Base URL: {base_url}, Default Timeout: {timeout}s")

    async def close(self):
        """Closes the underlying HTTP client. Should be called on application shutdown."""
        await self._client.aclose()
        logger.info("APIExecutor's HTTP client closed.")

    def _construct_url(self, endpoint: str) -> str:
        """Constructs the full URL, prepending base_url if endpoint is relative."""
        if self.base_url and not endpoint.startswith(('http://', 'https://')):
            # Ensure no double slashes if base_url ends with / and endpoint starts with /
            return self.base_url.rstrip('/') + '/' + endpoint.lstrip('/')
        return endpoint

    async def execute_api(
        self,
        operationId: str, # For logging/tracing
        method: str,
        endpoint: str, # Should be the full path or relative to base_url
        payload: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        # path_params are assumed to be already substituted into the endpoint string by the caller
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single API call.

        Args:
            operationId (str): The operation ID for logging and context.
            method (str): The HTTP method (e.g., "GET", "POST", "PUT", "DELETE").
            endpoint (str): The API endpoint path.
            payload (Optional[Dict[str, Any]]): The JSON request body for POST/PUT/PATCH.
            query_params (Optional[Dict[str, Any]]): Query parameters for the URL.
            headers (Optional[Dict[str, Any]]): Custom headers for this specific request.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "status_code": HTTP status code of the response.
                - "response_body": Parsed JSON response body if 'application/json', otherwise raw text.
                - "response_headers": Dictionary of response headers.
                - "execution_time": Time taken for the API call in seconds (float).
                - "error": Error message if an exception occurred, otherwise None.
                - "request_url": The final URL that was called.
                - "request_method": The HTTP method used.
        """
        start_time = time.perf_counter()
        
        request_method = method.upper()
        final_url = self._construct_url(endpoint)
        
        # Prepare headers: start with default, update with custom, then specific request headers
        final_headers = self.default_headers.copy()
        if headers:
            # Ensure all header keys and values are strings, as httpx expects
            final_headers.update({str(k): str(v) for k, v in headers.items()})

        # Ensure Content-Type for relevant methods if payload exists and not already set
        if request_method in ["POST", "PUT", "PATCH"] and payload is not None:
            # Check for content-type in a case-insensitive way
            if not any(key.lower() == 'content-type' for key in final_headers.keys()):
                final_headers['Content-Type'] = 'application/json'
        
        # Logging request details
        # MODIFIED: Payload logging is now conditional on DEBUG level for security.
        payload_log_message = "Payload: [Omitted in INFO, available in DEBUG]"
        if logger.isEnabledFor(logging.DEBUG):
            log_payload_preview = str(payload)[:200] + "..." if payload and len(str(payload)) > 200 else payload
            payload_log_message = f"Payload Preview: {log_payload_preview}"

        logger.info(f"Executing API call for OpID '{operationId}': {request_method} {final_url}")
        logger.debug(
            f"OpID '{operationId}' - Query Params: {query_params}, Headers: {final_headers}, {payload_log_message}"
        )


        response_data = {
            "status_code": None,
            "response_body": None,
            "response_headers": None,
            "execution_time": None,
            "error": None,
            "request_url": final_url,
            "request_method": request_method,
        }

        try:
            request_kwargs = {
                "method": request_method,
                "url": final_url,
                "params": query_params, # httpx handles query params correctly
                "headers": final_headers,
            }
            # Add payload if method requires it and payload is provided
            if request_method in ["POST", "PUT", "PATCH", "DELETE"] and payload is not None:
                # Check content type from final_headers (case-insensitive)
                current_content_type = ""
                for k, v in final_headers.items():
                    if k.lower() == 'content-type':
                        current_content_type = v.lower()
                        break
                
                if 'application/json' in current_content_type:
                    request_kwargs["json"] = payload # httpx handles JSON serialization
                else: # For other content types like form data (not fully handled here, assumes string/bytes)
                    request_kwargs["data"] = payload


            http_response: httpx.Response = await self._client.request(**request_kwargs)

            response_data["status_code"] = http_response.status_code
            response_data["response_headers"] = dict(http_response.headers) # Convert Headers object to dict

            # Attempt to parse JSON, otherwise get raw text
            content_type = http_response.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                try:
                    response_data["response_body"] = http_response.json()
                except json.JSONDecodeError:
                    logger.warning(f"OpID '{operationId}': Failed to decode JSON response despite content-type. Status: {http_response.status_code}. Raw text preview: {http_response.text[:200]}...")
                    response_data["response_body"] = http_response.text # Store raw text if JSON parsing fails
            else:
                response_data["response_body"] = http_response.text
            
            # Optionally, raise an exception for bad status codes (4xx or 5xx) to be caught by the generic handler below.
            # http_response.raise_for_status() # Uncomment if you want to handle all HTTP errors this way.

            # Log warning for non-2xx status codes if not raising exception above
            if not (200 <= http_response.status_code < 300):
                 logger.warning(f"OpID '{operationId}': Received non-2xx status: {http_response.status_code}. Response preview: {str(response_data['response_body'])[:200]}...")
                 # You might want to set response_data["error"] here based on status
                 # For now, the calling workflow_executor or manager checks status_code.

        except httpx.TimeoutException as e_timeout:
            logger.error(f"OpID '{operationId}': Timeout during API call to {final_url}: {e_timeout}")
            response_data["error"] = f"Timeout: {str(e_timeout)}"
            response_data["status_code"] = 408 # Request Timeout (conceptual, httpx might not set this)
        except httpx.RequestError as e_request: # Catches connection errors, invalid URLs etc.
            logger.error(f"OpID '{operationId}': Request error during API call to {final_url}: {e_request}")
            response_data["error"] = f"Request Error: {str(e_request)}"
            # Conceptual status code for service unavailable due to request issues
            response_data["status_code"] = 503 if response_data["status_code"] is None else response_data["status_code"]
        except Exception as e_general: # Catch any other unexpected errors
            logger.critical(f"OpID '{operationId}': Unexpected error during API call to {final_url}: {e_general}", exc_info=True)
            response_data["error"] = f"Unexpected Error: {str(e_general)}"
            # Conceptual status code for internal server error
            response_data["status_code"] = 500 if response_data["status_code"] is None else response_data["status_code"]
        finally:
            end_time = time.perf_counter()
            response_data["execution_time"] = round(end_time - start_time, 4)
            logger.info(f"OpID '{operationId}': Finished API call. Status: {response_data['status_code']}, Time: {response_data['execution_time']:.4f}s")

        return response_data

# Example Usage (for testing this file directly, not part of the main app flow)
async def main_test():
    # This is a dummy base URL for a publicly available test API
    # For your actual use, you'd configure this based on the OpenAPI spec's servers block
    # or expect absolute URLs in the node definitions.
    
    # Configure logging to see DEBUG messages for payload preview
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with a public API
    executor = APIExecutor(base_url="https://jsonplaceholder.typicode.com")

    # Test GET
    get_result = await executor.execute_api(
        operationId="getPosts",
        method="GET",
        endpoint="/posts/1",
    )
    print("\n--- GET Test Result ---")
    print(json.dumps(get_result, indent=2))

    # Test POST
    post_payload = {"title": "foo", "body": "bar", "userId": 1}
    post_result = await executor.execute_api(
        operationId="createPost",
        method="POST",
        endpoint="/posts",
        payload=post_payload,
        headers={"X-Custom-Header": "TestValue"}
    )
    print("\n--- POST Test Result ---")
    print(json.dumps(post_result, indent=2))

    # Test Not Found
    not_found_result = await executor.execute_api(
        operationId="getNonExistent",
        method="GET",
        endpoint="/nonexistentpath/123"
    )
    print("\n--- Not Found Test Result ---")
    print(json.dumps(not_found_result, indent=2))

    await executor.close() # Important to close the client

if __name__ == "__main__":
    # To run the test: python api_executor.py
    # Ensure logging is configured to DEBUG to see payload previews in test.
    # asyncio.run(main_test()) # This line is commented out to prevent execution when imported.
    # If you want to test, uncomment the line above and run `python api_executor.py` from your terminal.
    pass
