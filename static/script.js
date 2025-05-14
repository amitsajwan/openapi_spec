// static/script.js

// Ensure this runs after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const specInput = document.getElementById('specInput');
    const specUrlInput = document.getElementById('specUrl');
    const specFileInput = document.getElementById('specFile');
    const submitSpecButton = document.getElementById('submitSpecButton');
    const chatMessages = document.getElementById('chatMessages');
    const graphContainer = document.getElementById('graphContainer');
    const userInput = document.getElementById('userInput');
    const sendMessageButton = document.getElementById('sendMessageButton');
    const clearChatButton = document.getElementById('clearChatButton');
    const thinkingIndicatorContainer = document.getElementById('thinkingIndicatorContainer');
    const thinkingIndicatorText = document.getElementById('thinkingIndicatorText');
    const apiResponseContainer = document.getElementById('apiResponseContainer');
    const apiResponseContent = document.getElementById('apiResponseContent');
    const closeApiResponseButton = document.getElementById('closeApiResponse');

    let ws;
    let isThinking = false; // Global state for thinking indicator
    let currentGraph = null; // To store the current graph data for re-rendering

    // Function to show the thinking indicator
    function startThinking(message = "Processing...") {
        isThinking = true;
        if (thinkingIndicatorText) thinkingIndicatorText.textContent = message;
        if (thinkingIndicatorContainer) {
            thinkingIndicatorContainer.classList.remove('opacity-0', 'pointer-events-none');
            thinkingIndicatorContainer.classList.add('opacity-100');
        }
        if (submitSpecButton) submitSpecButton.disabled = true;
        if (sendMessageButton) sendMessageButton.disabled = true;
        // Optionally add a message to chat, but the indicator itself is primary
        // addMessageToChat('status', message); 
        console.log("UI: Thinking started - ", message);
    }

    // Function to hide the thinking indicator
    function stopThinking(message = "Done.") {
        isThinking = false;
        if (thinkingIndicatorContainer) {
            thinkingIndicatorContainer.classList.add('opacity-0', 'pointer-events-none');
            thinkingIndicatorContainer.classList.remove('opacity-100');
        }
        if (submitSpecButton) submitSpecButton.disabled = false;
        if (sendMessageButton) sendMessageButton.disabled = false;
        // Optionally add a status message to chat
        // addMessageToChat('status', message);
        console.log("UI: Thinking stopped - ", message);
    }
    
    function connectWebSocket() {
        // Determine WebSocket protocol (ws or wss)
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/openai_spec_agent`;
        
        ws = new WebSocket(wsUrl);

        ws.onopen = function(event) {
            console.log("WebSocket connection established.");
            addMessageToChat('status', "Connected to the agent.");
            stopThinking("Connected."); // Ensure UI is reset on new connection
        };

        ws.onmessage = function(event) {
            try {
                const parsed_event_data = JSON.parse(event.data);
                console.log("WebSocket message received:", parsed_event_data);
                
                const messageType = parsed_event_data.type;
                const data = parsed_event_data.data || {}; // Ensure data object exists

                switch (messageType) {
                    case 'status_update':
                        if (data.event === 'thinking_started') {
                            startThinking(data.message || "Processing...");
                        } else if (data.event === 'thinking_finished') {
                            stopThinking(data.message || "Processing finished.");
                        } else {
                             addMessageToChat('status', `Status: ${data.event} - ${data.message || ''}`);
                        }
                        break;
                    case 'intermediate_message':
                        addMessageToChat(data.source || 'assistant', data.message);
                        if (isThinking && data.message && thinkingIndicatorText) {
                            // Update thinking indicator text with the latest intermediate message
                            thinkingIndicatorText.textContent = data.message.length > 50 ? data.message.substring(0, 47) + "..." : data.message;
                        }
                        break;
                    case 'graph_update':
                        if (data.graph) {
                            currentGraph = data.graph; // Store the graph data
                            renderGraph(currentGraph);
                            addMessageToChat('status', "Workflow graph updated.");
                        }
                        break;
                    case 'api_response':
                        displayApiResponse(data);
                        // Typically, an api_response might come after "thinking" has finished for that specific call.
                        // If it's part of a larger flow, the overall "thinking" might still be active.
                        break;
                    case 'final_response':
                        addMessageToChat('assistant', data.message);
                        if (data.result) { // Example: if final response has structured result
                            // You might want to display this structured result differently
                            console.log("Final result data:", data.result);
                        }
                        stopThinking(data.message || "Task complete."); // Ensure thinking stops
                        break;
                    case 'error':
                        addMessageToChat('error', `Error: ${data.message}`);
                        stopThinking("An error occurred."); // Stop thinking on error
                        break;
                    default:
                        addMessageToChat('unknown', `Received: ${event.data}`);
                }
            } catch (e) {
                console.error("Error processing WebSocket message:", e);
                addMessageToChat('error', "Received an unparseable message from server.");
            }
        };

        ws.onclose = function(event) {
            console.log("WebSocket connection closed.", event.reason);
            addMessageToChat('status', "Connection closed. Attempting to reconnect in 5 seconds...");
            stopThinking("Connection lost."); // Reset UI
            setTimeout(connectWebSocket, 5000); // Attempt to reconnect
        };

        ws.onerror = function(error) {
            console.error("WebSocket error:", error);
            addMessageToChat('error', "WebSocket connection error.");
            // onclose will usually be called after onerror, so reconnection logic is there.
            // stopThinking() will also be called by onclose.
        };
    }

    function addMessageToChat(source, message, type = 'text') {
        if (!chatMessages) return;
        const messageElement = document.createElement('div');
        messageElement.classList.add('mb-2', 'p-3', 'rounded-lg', 'max-w-2xl', 'break-words', 'text-sm');

        let sourcePrefix = "";
        if (source === 'user') {
            messageElement.classList.add('bg-blue-500', 'text-white', 'self-end', 'ml-auto');
            sourcePrefix = '<strong class="font-semibold">You:</strong> ';
        } else if (source === 'assistant') {
            messageElement.classList.add('bg-gray-200', 'text-gray-800', 'self-start', 'mr-auto');
            sourcePrefix = '<strong class="font-semibold">Agent:</strong> ';
        } else if (source === 'status') {
            messageElement.classList.add('bg-yellow-100', 'text-yellow-700', 'self-center', 'text-xs', 'italic');
        } else if (source === 'error') {
            messageElement.classList.add('bg-red-100', 'text-red-700', 'self-start', 'mr-auto', 'font-semibold');
            sourcePrefix = '<strong class="font-semibold">Error:</strong> ';
        } else { // unknown or system
            messageElement.classList.add('bg-gray-50', 'text-gray-500', 'self-center', 'text-xs');
        }
        
        // Sanitize message before setting innerHTML if it can contain user-generated HTML
        // For now, assuming messages are text or controlled HTML from backend
        messageElement.innerHTML = sourcePrefix + message; 
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll
    }

    function sendWebSocketMessage(type, payload) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const message = JSON.stringify({ type: type, ...payload });
            ws.send(message);
            console.log("Sent to WS:", message);
        } else {
            addMessageToChat('error', "WebSocket is not connected. Cannot send message.");
            console.error("WebSocket is not connected.");
        }
    }

    if (submitSpecButton) {
        submitSpecButton.addEventListener('click', () => {
            let specData = specInput.value.trim();
            const url = specUrlInput.value.trim();
            const file = specFileInput.files[0];
            let source = 'text';

            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    specData = e.target.result;
                    source = 'file';
                    addMessageToChat('user', `Submitted OpenAPI Spec from file: ${file.name}`);
                    sendWebSocketMessage('process_openapi_spec', { openapi_spec_string: specData, source: source, file_name: file.name });
                };
                reader.onerror = function(e) {
                    addMessageToChat('error', 'Error reading file.');
                    console.error("File reading error:", e);
                };
                reader.readAsText(file);
            } else if (url) {
                specData = url; // The backend will fetch the URL
                source = 'url';
                addMessageToChat('user', `Submitted OpenAPI Spec from URL: ${url}`);
                sendWebSocketMessage('process_openapi_spec', { openapi_spec_url: specData, source: source });
            } else if (specData) {
                source = 'text';
                addMessageToChat('user', `Submitted OpenAPI Spec (text input)`);
                sendWebSocketMessage('process_openapi_spec', { openapi_spec_string: specData, source: source });
            } else {
                addMessageToChat('error', 'Please provide an OpenAPI spec (text, URL, or file).');
                return;
            }
            // "Thinking" will be started by the backend's response.
        });
    }

    if (sendMessageButton) {
        sendMessageButton.addEventListener('click', () => {
            const messageText = userInput.value.trim();
            if (messageText) {
                addMessageToChat('user', messageText);
                sendWebSocketMessage('user_interaction', { text: messageText, current_graph: currentGraph });
                userInput.value = '';
            }
        });
    }
    
    if (userInput) {
        userInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault(); // Prevent new line
                sendMessageButton.click();
            }
        });
    }

    if (clearChatButton) {
        clearChatButton.addEventListener('click', () => {
            if (chatMessages) chatMessages.innerHTML = '';
            if (graphContainer) graphContainer.innerHTML = ''; // Also clear graph
            currentGraph = null;
            if (apiResponseContainer) apiResponseContainer.classList.add('hidden');
            addMessageToChat('status', 'Chat and graph cleared.');
        });
    }
    
    if (closeApiResponseButton) {
        closeApiResponseButton.addEventListener('click', () => {
            if (apiResponseContainer) apiResponseContainer.classList.add('hidden');
        });
    }

    function displayApiResponse(data) {
        if (!apiResponseContainer || !apiResponseContent) return;
        
        let contentHtml = `<h3 class="text-lg font-semibold mb-2">${data.operation_id || 'API Response'}</h3>`;
        contentHtml += `<p class="text-sm mb-1"><span class="font-semibold">Status:</span> ${data.status_code || 'N/A'}</p>`;
        
        if (data.headers) {
            contentHtml += '<p class="text-sm mb-1"><span class="font-semibold">Headers:</span></p>';
            contentHtml += `<pre class="bg-gray-100 p-2 rounded text-xs overflow-x-auto max-h-32"><code>${JSON.stringify(data.headers, null, 2)}</code></pre>`;
        }
        if (data.body) {
            contentHtml += '<p class="text-sm mt-2 mb-1"><span class="font-semibold">Body:</span></p>';
            let bodyContent = data.body;
            try {
                // Try to parse if it's a JSON string, then re-stringify for pretty print
                const parsedBody = JSON.parse(bodyContent);
                bodyContent = JSON.stringify(parsedBody, null, 2);
            } catch (e) {
                // Not a JSON string, or already an object. Or just plain text.
                if (typeof bodyContent === 'object') {
                    bodyContent = JSON.stringify(bodyContent, null, 2);
                }
            }
            contentHtml += `<pre class="bg-gray-100 p-2 rounded text-xs overflow-x-auto max-h-60"><code>${escapeHtml(bodyContent)}</code></pre>`;
        }
        
        apiResponseContent.innerHTML = contentHtml;
        apiResponseContainer.classList.remove('hidden');
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            // If it's not a string (e.g., already an object from JSON.parse), stringify it
            try {
                unsafe = JSON.stringify(unsafe, null, 2);
            } catch (e) {
                return ''; // Or some error placeholder
            }
        }
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    function renderGraph(graphData) {
        if (!graphContainer || !cytoscape) {
            console.error("Graph container or Cytoscape library not found.");
            if (!cytoscape) addMessageToChat('error', "Cytoscape library not loaded. Cannot render graph.");
            return;
        }
        graphContainer.innerHTML = ''; // Clear previous graph

        try {
            const cy = cytoscape({
                container: graphContainer,
                elements: graphData.elements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'background-color': (ele) => ele.data('type') === 'llm_call' ? '#FF69B4' : (ele.data('type') === 'api_call' ? '#1E90FF' : (ele.data('type') === 'user_input' ? '#FFD700' : (ele.data('type') === 'knowledge_base' ? '#32CD32' : '#666'))),
                            'label': 'data(label)',
                            'width': 'label',
                            'height': 'label',
                            'padding': '12px', // Increased padding
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'shape': 'round-rectangle', // More modern shape
                            'font-size': '10px', // Smaller font for fitting more text
                            'color': '#fff', // White text for better contrast on dark backgrounds
                            'text-outline-width': 1, // Thin outline for text
                            'text-outline-color': '#333', // Dark outline
                            'border-width': 2,
                            'border-color': '#4A5568' // Darker border
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': '#9CA3AF', // Softer edge color
                            'target-arrow-color': '#9CA3AF',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier', // Smoother curves
                            'label': 'data(label)', // Edge labels if provided
                            'font-size': '8px',
                            'color': '#4A5568',
                            'text-rotation': 'autorotate',
                            'text-margin-y': -10
                        }
                    },
                    { // Style for selected nodes
                        selector: 'node:selected',
                        style: {
                            'border-width': 3,
                            'border-color': '#F59E0B' // Amber color for selection
                        }
                    }
                ],
                layout: {
                    name: 'dagre', // Directed graph layout
                    rankDir: 'TB', // Top-to-Bottom
                    spacingFactor: 1.2,
                    padding: 20
                }
            });

            // Pan and zoom controls (optional, if you want to add cytoscape.js-panzoom)
            // cy.panzoom({}); 

            // Tooltip for nodes
            cy.nodes().forEach(node => {
                let content = `<strong>ID:</strong> ${node.id()}<br/><strong>Type:</strong> ${node.data('type') || 'N/A'}`;
                if (node.data('description')) {
                    content += `<br/><strong>Desc:</strong> ${escapeHtml(node.data('description'))}`;
                }
                if(node.data('status')) {
                    content += `<br/><strong>Status:</strong> ${node.data('status')}`;
                }
                // Create a tippy instance for each node
                // Ensure Tippy.js is loaded for this to work
                if (typeof tippy === 'function') {
                    tippy(node.popperRef(), {
                        content: content,
                        trigger: 'manual', // We'll show/hide manually or on hover
                        allowHTML: true,
                        placement: 'top',
                        arrow: true,
                        interactive: true, // Allows clicking links in tooltip
                        theme: 'light-border', // Or your custom theme
                        onShow(instance) {
                            // You can add custom logic when tooltip shows
                        },
                    }).show(); // Show by default, or use mouseover/mouseout
                    
                    // Example: Show on mouseover, hide on mouseout
                    let tip = node.scratch().tippy; // Get the tippy instance if stored
                    if (!tip && typeof tippy === 'function') {
                         tip = tippy(node.popperRef(), { /* ... options ... */ });
                         node.scratch().tippy = tip; // Store it
                    }
                    
                    node.on('mouseover', (event) => event.target.scratch().tippy && event.target.scratch().tippy.show());
                    node.on('mouseout', (event) => event.target.scratch().tippy && event.target.scratch().tippy.hide());
                    node.on('tap', function(event) {
                        // Handle node click, e.g., show details in a side panel
                        console.log('Clicked node:', event.target.data());
                        addMessageToChat('status', `Node clicked: ${event.target.data('label') || event.target.id()}`);
                        // You could also send this node's data to the backend for more actions
                        // sendWebSocketMessage('node_interaction', { node_id: event.target.id(), node_data: event.target.data() });
                    });

                } else {
                    console.warn("Tippy.js not loaded, tooltips will not be available for graph nodes.");
                }
            });
             // Fit graph to view with padding
            cy.fit(null, 30); // 30px padding

        } catch (e) {
            console.error("Error rendering graph:", e);
            addMessageToChat('error', "Failed to render workflow graph.");
            if (graphContainer) graphContainer.innerHTML = '<p class="text-red-500">Error rendering graph. Check console.</p>';
        }
    }
    
    // Initial connection
    connectWebSocket();
});
