// static/script.js

// Ensure this runs after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const specInput = document.getElementById('specInput');
    const specUrlInput = document.getElementById('specUrl');
    const specFileInput = document.getElementById('specFile');
    const submitSpecButton = document.getElementById('submitSpecButton');
    const chatMessages = document.getElementById('chatMessages');
    const graphContainer = document.getElementById('graphContainer'); // Assuming this is for Cytoscape
    const userInput = document.getElementById('userInput');
    const sendMessageButton = document.getElementById('sendMessageButton');
    const clearChatButton = document.getElementById('clearChatButton');
    const thinkingIndicatorContainer = document.getElementById('thinkingIndicatorContainer');
    const thinkingIndicatorText = document.getElementById('thinkingIndicatorText');
    const apiResponseContainer = document.getElementById('apiResponseContainer');
    const apiResponseContent = document.getElementById('apiResponseContent');
    const closeApiResponseButton = document.getElementById('closeApiResponse');
    const runWorkflowButton = document.getElementById('runWorkflowButton'); // From your HTML

    let ws;
    let isThinking = false; // Global state for thinking indicator
    let currentGraph = null; // To store the current graph data for re-rendering

    // Function to show the thinking indicator
    function startThinking(message = "Processing...") {
        if (isThinking) return;
        isThinking = true;
        if (thinkingIndicatorText) thinkingIndicatorText.textContent = message;
        if (thinkingIndicatorContainer) {
            thinkingIndicatorContainer.classList.remove('opacity-0', 'pointer-events-none');
            thinkingIndicatorContainer.classList.add('opacity-100');
        }
        if (submitSpecButton) submitSpecButton.disabled = true;
        if (sendMessageButton) sendMessageButton.disabled = true;
        if (runWorkflowButton) runWorkflowButton.disabled = true;
        console.log("UI: Thinking started - ", message);
    }

    // Function to hide the thinking indicator
    function stopThinking(message = "Done.") {
        if (!isThinking) return;
        isThinking = false;
        if (thinkingIndicatorContainer) {
            thinkingIndicatorContainer.classList.add('opacity-0', 'pointer-events-none');
            thinkingIndicatorContainer.classList.remove('opacity-100');
        }
        // Re-enable buttons based on whether a spec/graph is loaded
        const specOrGraphLoaded = currentGraph !== null; 
        if (submitSpecButton) submitSpecButton.disabled = false;
        if (sendMessageButton) sendMessageButton.disabled = !specOrGraphLoaded;
        if (runWorkflowButton) runWorkflowButton.disabled = !specOrGraphLoaded;
        console.log("UI: Thinking stopped - ", message);
    }
    
    function connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // The user's script.js uses this path. If session ID is needed, backend should handle it.
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/openapi_agent`; 
        
        ws = new WebSocket(wsUrl);

        ws.onopen = function(event) {
            console.log("WebSocket connection established.");
            addMessageToChat('status', "Connected to the agent.", "connection_status");
            stopThinking("Connected.");
            // Initial button states
            if (sendMessageButton) sendMessageButton.disabled = true;
            if (runWorkflowButton) runWorkflowButton.disabled = true;
        };

        ws.onmessage = function(event) {
            try {
                const parsed_event_data = JSON.parse(event.data);
                console.log("WebSocket message received:", parsed_event_data);
                
                const messageType = parsed_event_data.type; // This is the WebSocket message type
                const data = parsed_event_data.content || parsed_event_data.data || {}; // Accommodate different backend structures
                const source = parsed_event_data.source || 'assistant'; // Backend might specify source ('graph1_planning', 'graph2_execution', 'system')

                switch (messageType) {
                    // This case handles the thinking indicator based on explicit backend signals
                    // If you switch to client-managed thinking, this case would be removed/changed.
                    case 'status_update': // This was for backend-controlled thinking
                        if (data.event === 'thinking_started') {
                            startThinking(data.message || "Processing...");
                        } else if (data.event === 'thinking_finished') {
                            stopThinking(data.message || "Processing finished.");
                        } else {
                             addMessageToChat('status', `Status: ${data.event} - ${data.message || ''}`, messageType);
                        }
                        break;
                    
                    // These are message types from your backend's send_websocket_message_helper
                    case 'status': // General status from backend
                    case 'info':
                    case 'warning':
                        addMessageToChat(source, data.message || 'Status update.', messageType);
                        break;
                    case 'intermediate': // Intermediate messages from Graph 1
                    case 'intermediate_message': // For compatibility with previous naming
                        addMessageToChat(source, data.message, messageType);
                        if (isThinking && data.message && thinkingIndicatorText) {
                            thinkingIndicatorText.textContent = data.message.length > 70 ? data.message.substring(0, 67) + "..." : data.message;
                        }
                        break;
                    case 'graph_update':
                        if (data.graph || data.nodes || data.elements) { // data itself might be the graph object
                            currentGraph = data.graph || data; // data might be the graph object directly
                            renderGraph(currentGraph);
                            addMessageToChat('status', "Workflow graph updated.", messageType);
                            if (sendMessageButton) sendMessageButton.disabled = false;
                            if (runWorkflowButton) runWorkflowButton.disabled = false;
                        } else {
                            console.warn("Received graph_update without graph data:", data);
                        }
                        break;
                    case 'api_response':
                        displayApiResponse(data);
                        break;
                    case 'final': // Final response from Graph 1
                    case 'final_response': // For compatibility
                        addMessageToChat(source, data.message || "Processing complete.", messageType);
                        if (data.graph) { 
                            currentGraph = data.graph;
                            renderGraph(currentGraph);
                        }
                        // This stopThinking call is for backend-controlled thinking.
                        // If client-managed, it would be handled differently.
                        stopThinking(data.message || "Task complete."); 
                        if (currentGraph) {
                            if (sendMessageButton) sendMessageButton.disabled = false;
                            if (runWorkflowButton) runWorkflowButton.disabled = false;
                        }
                        break;
                    case 'error':
                        addMessageToChat('error', `Error: ${data.error || data.message || 'Unknown error.'}`, messageType);
                        // This stopThinking call is for backend-controlled thinking.
                        stopThinking("An error occurred."); 
                        break;
                    
                    // Graph 2 specific events (as per your websocket_helpers.py)
                    case 'human_intervention_required':
                        addMessageToChat(source, data.message || `Action required for node ${data.node_id}`, messageType);
                        // Potentially show modal here based on data
                        break;
                    case 'execution_update': // Generic update from Graph 2
                    case 'tool_start':
                    case 'tool_end':
                    case 'llm_start':
                    case 'llm_stream':
                    case 'llm_end':
                         addMessageToChat(source, data.message || `Execution update: ${messageType}`, messageType);
                         if (isThinking && data.message && thinkingIndicatorText) {
                            thinkingIndicatorText.textContent = data.message.length > 70 ? data.message.substring(0, 67) + "..." : data.message;
                        }
                         break;
                    case 'execution_completed':
                    case 'execution_failed':
                    case 'workflow_timeout':
                        addMessageToChat(source, data.message || `Workflow ${messageType}.`, messageType);
                        // This stopThinking call is for backend-controlled thinking.
                        stopThinking(`Workflow ${messageType}.`);
                        break;

                    default:
                        addMessageToChat('unknown', `Received unhandled type '${messageType}': ${(event.data || "").substring(0,150)}...`, messageType);
                }
            } catch (e) {
                console.error("Error processing WebSocket message:", e, "Raw data:", event.data);
                addMessageToChat('error', "Received an unparseable message from server.", "parse_error");
                // This stopThinking call is for backend-controlled thinking.
                stopThinking("Error processing server message.");
            }
        };

        ws.onclose = function(event) {
            console.log("WebSocket connection closed.", event.reason, "Code:", event.code);
            addMessageToChat('status', "Connection closed. Attempting to reconnect in 5 seconds...", "connection_status");
            stopThinking("Connection lost."); 
            isThinking = false; 
            setTimeout(connectWebSocket, 5000); 
        };

        ws.onerror = function(error) {
            console.error("WebSocket error:", error);
            addMessageToChat('error', "WebSocket connection error. Check console for details.", "ws_error");
        };
    }

    function addMessageToChat(source, message, wsMessageType, isHtml = false) {
        if (!chatMessages) return;

        const lastMessageElement = chatMessages.lastElementChild;
        let appendToLast = false;

        if (lastMessageElement &&
            lastMessageElement.dataset.source === source &&
            lastMessageElement.dataset.wsMessageType === wsMessageType &&
            (source === 'assistant' || source === 'graph1_planning' || source === 'graph2_execution') && // Only append for these sources
            (wsMessageType === 'intermediate' || wsMessageType === 'intermediate_message' || wsMessageType === 'llm_stream' || wsMessageType === 'status' || wsMessageType === 'execution_update') // Only for these types
           ) {
            appendToLast = true;
        }
        
        // Sanitize message if it's not explicitly HTML
        let sanitizedMessage = message;
        if (!isHtml) {
            const tempDiv = document.createElement('div');
            tempDiv.textContent = message; // Ensures text is treated as text
            sanitizedMessage = tempDiv.innerHTML.replace(/\n/g, '<br>'); // Convert newlines to <br>
        }


        if (appendToLast && lastMessageElement) {
            // Find the content span within the last message element to append to
            const contentSpan = lastMessageElement.querySelector('.message-text-content');
            if (contentSpan) {
                contentSpan.innerHTML += '<br>' + sanitizedMessage;
            } else { 
                // Fallback if no specific content span, append to whole innerHTML (less ideal)
                lastMessageElement.innerHTML += '<br>' + sanitizedMessage;
            }
        } else {
            const messageElement = document.createElement('div');
            messageElement.classList.add('mb-2', 'p-3', 'rounded-lg', 'max-w-2xl', 'break-words', 'text-sm', 'shadow-sm');
            messageElement.dataset.source = source;
            messageElement.dataset.wsMessageType = wsMessageType; // Store the WebSocket message type

            let sourcePrefix = "";
            let prefixClasses = "font-semibold message-sender-tag";

            if (source === 'user') {
                messageElement.classList.add('bg-blue-500', 'text-white', 'self-end', 'ml-auto');
                sourcePrefix = 'You:';
            } else if (source === 'assistant' || source === 'graph1_planning' || source === 'graph2_execution' || source === 'agent') {
                messageElement.classList.add('bg-gray-200', 'text-gray-800', 'self-start', 'mr-auto');
                sourcePrefix = 'Agent:';
                 if(source === 'graph1_planning') sourcePrefix = 'Planner:';
                 if(source === 'graph2_execution') sourcePrefix = 'Executor:';
            } else if (source === 'status' || source === 'system' || source === 'system_critical' || source === 'system_warning') {
                messageElement.classList.add('bg-yellow-100', 'text-yellow-800', 'self-center', 'text-xs', 'italic', 'text-center', 'py-1', 'px-2');
                sourcePrefix = 'Status:';
                if(source === 'system_critical') messageElement.classList.replace('bg-yellow-100','bg-red-200'); messageElement.classList.replace('text-yellow-800','text-red-800');
            } else if (source === 'error' || source === 'system_error') {
                messageElement.classList.add('bg-red-100', 'text-red-700', 'self-start', 'mr-auto', 'font-semibold');
                sourcePrefix = 'Error:';
            } else { // unknown
                messageElement.classList.add('bg-gray-50', 'text-gray-500', 'self-center', 'text-xs', 'italic');
                sourcePrefix = 'System:';
            }
            
            messageElement.innerHTML = `<strong class="${prefixClasses}">${sourcePrefix}</strong> <span class="message-text-content">${sanitizedMessage}</span>`;
            chatMessages.appendChild(messageElement);
        }
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendWebSocketMessage(type, payload) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            // This script version relies on backend to send thinking_started/finished
            // If you want client-managed thinking, call startThinking() here.
            // startThinking("Sending request..."); 

            const message = JSON.stringify({ type: type, ...payload }); // Ensure payload is spread if it's an object
            ws.send(message);
            console.log("Sent to WS:", message);
        } else {
            addMessageToChat('error', "WebSocket is not connected. Cannot send message.", "ws_error_send");
            console.error("WebSocket is not connected.");
            // stopThinking("Connection error."); // If client-managed thinking
        }
    }

    if (submitSpecButton) {
        submitSpecButton.addEventListener('click', () => {
            let specData = specInput.value.trim();
            const url = specUrlInput.value.trim();
            const file = specFileInput.files[0];
            let source_type = 'text'; // Renamed from 'source' to avoid conflict
            let userMessage = "";

            // Prepare the payload for the backend
            let payload = { source: source_type }; // Default source

            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    specData = e.target.result;
                    payload.source = 'file';
                    payload.openapi_spec_string = specData;
                    payload.file_name = file.name;
                    userMessage = `Submitted OpenAPI Spec from file: ${file.name}`;
                    addMessageToChat('user', userMessage, "user_spec_submission");
                    sendWebSocketMessage('process_openapi_spec', payload);
                };
                reader.onerror = function(e) {
                    addMessageToChat('error', 'Error reading file.', "file_error");
                    console.error("File reading error:", e);
                };
                reader.readAsText(file);
            } else if (url) {
                payload.source = 'url';
                payload.openapi_spec_url = url; // Backend will fetch
                userMessage = `Submitted OpenAPI Spec from URL: ${url}`;
                addMessageToChat('user', userMessage, "user_spec_submission");
                sendWebSocketMessage('process_openapi_spec', payload);
            } else if (specData) {
                payload.source = 'text';
                payload.openapi_spec_string = specData;
                userMessage = `Submitted OpenAPI Spec (text input)`;
                addMessageToChat('user', userMessage, "user_spec_submission");
                sendWebSocketMessage('process_openapi_spec', payload);
            } else {
                addMessageToChat('error', 'Please provide an OpenAPI spec (text, URL, or file).', "input_error");
                return;
            }
        });
    }

    if (sendMessageButton) {
        sendMessageButton.addEventListener('click', () => {
            const messageText = userInput.value.trim();
            if (messageText) {
                addMessageToChat('user', messageText, "user_interaction");
                // Pass current_graph if your backend needs it for context
                // The backend message structure for 'user_interaction' might expect 'text' and 'current_graph' in a 'content' object.
                // Adjust based on your backend's send_websocket_message_helper and how it structures the received JSON.
                // For now, assuming backend directly uses top-level keys from the payload.
                sendWebSocketMessage('user_interaction', { 
                    text: messageText, 
                    current_graph_elements: currentGraph ? currentGraph.elements : null, // Or send full currentGraph if needed
                    // The backend's handle_websocket_connection and _initialize_bot_state_for_turn
                    // will need to correctly parse this.
                });
                userInput.value = '';
            }
        });
    }
    
    if (userInput) {
        userInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault(); 
                sendMessageButton.click();
            }
        });
    }

    if (clearChatButton) {
        clearChatButton.addEventListener('click', () => {
            if (chatMessages) chatMessages.innerHTML = '';
            if (graphContainer) graphContainer.innerHTML = ''; 
            currentGraph = null;
            if (apiResponseContainer) apiResponseContainer.classList.add('hidden');
            addMessageToChat('status', 'Chat and graph cleared.', "ui_action");
            if(sendMessageButton) sendMessageButton.disabled = true;
            if(runWorkflowButton) runWorkflowButton.disabled = true;
        });
    }
    
    if (closeApiResponseButton) {
        closeApiResponseButton.addEventListener('click', () => {
            if (apiResponseContainer) apiResponseContainer.classList.add('hidden');
        });
    }
    
    // --- runCurrentWorkflow ---
    // Make it globally accessible if called by onclick, or add event listener
    window.runCurrentWorkflow = function() {
        if (!currentGraph || !currentGraph.elements) { // Check if graph and its elements exist
            addMessageToChat('error', "No workflow graph is loaded or graph is empty.", "workflow_error");
            return;
        }
        addMessageToChat('user', "Attempting to run the current workflow...", "user_action");
        // The backend's handle_websocket_connection will receive this.
        // It needs to route it appropriately, likely to Graph 1 to set up Graph 2.
        // The payload should match what _initialize_bot_state_for_turn and the router expect.
        sendWebSocketMessage('user_interaction', { // Using 'user_interaction' to go through the main router
            text: "run workflow", // User input that the router can understand
            // current_graph: currentGraph, // Send the whole graph object if needed by the router/planner
            // For simplicity, the router might just look for the "run workflow" text.
            // If Graph 1 needs the graph data to initiate Graph 2, ensure it's available in BotState.
        });
    };
    // If runWorkflowButton exists, you could also do:
    // if (runWorkflowButton) {
    // runWorkflowButton.addEventListener('click', runCurrentWorkflow);
    // }


    function displayApiResponse(data) {
        if (!apiResponseContainer || !apiResponseContent) return;
        
        let contentHtml = `<h3 class="text-lg font-semibold mb-2">${data.operation_id || data.node_id || 'API Response'}</h3>`; // Use node_id if operation_id missing
        contentHtml += `<p class="text-sm mb-1"><span class="font-semibold">Status:</span> ${data.status_code || 'N/A'}</p>`;
        
        if (data.headers) {
            contentHtml += '<p class="text-sm mb-1"><span class="font-semibold">Headers:</span></p>';
            contentHtml += `<pre class="bg-gray-100 p-2 rounded text-xs overflow-x-auto max-h-32"><code>${escapeHtml(JSON.stringify(data.headers, null, 2))}</code></pre>`;
        }
        if (data.body || data.error) { // Check for body or error
            contentHtml += `<p class="text-sm mt-2 mb-1"><span class="font-semibold">${data.error ? 'Error Body:' : 'Body:'}</span></p>`;
            let bodyContent = data.body || data.error; // Display error if body is not present
            
            // Attempt to parse and pretty-print if JSON, otherwise display as is
            try {
                const parsedBody = (typeof bodyContent === 'string') ? JSON.parse(bodyContent) : bodyContent;
                bodyContent = JSON.stringify(parsedBody, null, 2);
            } catch (e) {
                // Not a JSON string, or already an object that couldn't be stringified well.
                // Or just plain text. Keep bodyContent as is.
                if (typeof bodyContent !== 'string') { // Ensure it's a string for escapeHtml
                   bodyContent = String(bodyContent);
                }
            }
            contentHtml += `<pre class="bg-gray-100 p-2 rounded text-xs overflow-x-auto max-h-60"><code>${escapeHtml(bodyContent)}</code></pre>`;
        }
        
        apiResponseContent.innerHTML = contentHtml;
        apiResponseContainer.classList.remove('hidden');
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            try {
                unsafe = String(unsafe); // Convert to string if not already
            } catch (e) {
                return ''; 
            }
        }
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    function renderGraph(graphData) { // graphData is expected to be the object from backend (e.g., GraphOutput.model_dump())
        if (!graphContainer || !window.cytoscape) {
            console.error("Graph container or Cytoscape library not found.");
            if (!window.cytoscape) addMessageToChat('error', "Cytoscape library not loaded. Cannot render graph.", "graph_error");
            return;
        }
        graphContainer.innerHTML = ''; 

        try {
            let elementsToRender;
            if (graphData && graphData.nodes && graphData.edges) { // Common structure for Cytoscape elements
                elementsToRender = { nodes: graphData.nodes, edges: graphData.edges };
            } else if (graphData && graphData.elements) { // If backend sends it under 'elements' key
                 elementsToRender = graphData.elements;
            } else {
                console.warn("Graph data is missing 'nodes'/'edges' or 'elements'. Attempting to use graphData directly if it's an array (legacy).", graphData);
                // Fallback for older structures or direct element arrays, though less likely with Pydantic models
                elementsToRender = Array.isArray(graphData) ? graphData : []; 
                if (!Array.isArray(graphData) || graphData.length === 0) {
                     addMessageToChat('status', 'Graph data is empty or in an unexpected format.', 'graph_status');
                }
            }
            
            if (!elementsToRender || (!elementsToRender.nodes && !elementsToRender.edges && !Array.isArray(elementsToRender)) ) {
                logger.error("No valid elements found to render graph.");
                addMessageToChat('status', 'No graph elements to display.', 'graph_status');
                return;
            }


            const cy = window.cytoscape({
                container: graphContainer,
                elements: elementsToRender,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'background-color': (ele) => {
                                const type = ele.data('type');
                                if (type === 'llm_call') return '#FF69B4'; 
                                if (type === 'api_call') return '#1E90FF'; 
                                if (type === 'user_input') return '#FFD700'; 
                                if (type === 'knowledge_base') return '#32CD32'; 
                                if (type === 'start_node') return '#90EE90'; 
                                if (type === 'end_node') return '#FA8072'; 
                                return '#666'; 
                            },
                            'label': 'data(label)',
                            'width': 'label', 'height': 'label', 'padding': '12px', 
                            'text-valign': 'center', 'text-halign': 'center',
                            'shape': 'round-rectangle', 'font-size': '10px', 
                            'color': '#fff', 'text-outline-width': 1, 'text-outline-color': '#333', 
                            'border-width': 2, 'border-color': '#4A5568' 
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2, 'line-color': '#9CA3AF', 
                            'target-arrow-color': '#9CA3AF', 'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier', 'label': 'data(label)', 
                            'font-size': '8px', 'color': '#4A5568',
                            'text-rotation': 'autorotate', 'text-margin-y': -10
                        }
                    },
                    { 
                        selector: 'node:selected',
                        style: { 'border-width': 3, 'border-color': '#F59E0B' }
                    }
                ],
                layout: {
                    name: 'dagre', rankDir: 'TB', spacingFactor: 1.3, padding: 30
                }
            });

            cy.nodes().forEach(node => {
                let content = `<strong>ID:</strong> ${node.id()}<br/><strong>Type:</strong> ${node.data('type') || 'N/A'}`;
                if (node.data('description')) {
                    content += `<br/><strong>Desc:</strong> ${escapeHtml(node.data('description'))}`;
                }
                if(node.data('status')) { // If nodes have status from backend
                    content += `<br/><strong>Status:</strong> ${node.data('status')}`;
                }
                
                if (typeof tippy === 'function') {
                    const tippyInstance = tippy(node.popperRef(), { // node.popperRef() is correct
                        content: content, trigger: 'manual', allowHTML: true,
                        placement: 'top', arrow: true, interactive: true, 
                        theme: 'light-border', 
                    });
                    node.scratch('tippy', tippyInstance); 
                    
                    node.on('mouseover', (event) => { const tip = event.target.scratch('tippy'); if(tip) tip.show(); });
                    node.on('mouseout', (event) => { const tip = event.target.scratch('tippy'); if(tip) tip.hide(); });
                    node.on('tap', function(event) {
                        console.log('Clicked node:', event.target.data());
                        addMessageToChat('status', `Node clicked: ${event.target.data('label') || event.target.id()}`, "node_event");
                    });

                } else if (!window.tippyWarningShown) {
                    console.warn("Tippy.js not loaded, tooltips will not be available for graph nodes.");
                    window.tippyWarningShown = true; 
                }
            });
            cy.fit(null, 30); 

        } catch (e) {
            console.error("Error rendering graph:", e, "Graph data used:", graphData);
            addMessageToChat('error', "Failed to render workflow graph. Check console.", "graph_error");
            if (graphContainer) graphContainer.innerHTML = '<p class="text-red-500 p-4">Error rendering graph. Please check the console for details.</p>';
        }
    }
    
    connectWebSocket();
});
