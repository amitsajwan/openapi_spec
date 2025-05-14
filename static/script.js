// static/script.js

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
    const runWorkflowButton = document.getElementById('runWorkflowButton'); // Added

    let ws;
    let isThinking = false; // Global state for thinking indicator
    let currentGraph = null; // To store the current graph data for re-rendering

    // Function to show the thinking indicator
    function startThinking(message = "Processing...") {
        if (isThinking) return; // Avoid multiple starts if already thinking
        isThinking = true;
        if (thinkingIndicatorText) thinkingIndicatorText.textContent = message;
        if (thinkingIndicatorContainer) {
            thinkingIndicatorContainer.classList.remove('opacity-0', 'pointer-events-none');
            thinkingIndicatorContainer.classList.add('opacity-100');
        }
        if (submitSpecButton) submitSpecButton.disabled = true;
        if (sendMessageButton) sendMessageButton.disabled = true;
        if (runWorkflowButton) runWorkflowButton.disabled = true; // Disable run workflow button too
        console.log("UI: Thinking started - ", message);
    }

    // Function to hide the thinking indicator
    function stopThinking(message = "Done.") {
        if (!isThinking) return; // Avoid multiple stops
        isThinking = false;
        if (thinkingIndicatorContainer) {
            thinkingIndicatorContainer.classList.add('opacity-0', 'pointer-events-none');
            thinkingIndicatorContainer.classList.remove('opacity-100');
        }
        // Re-enable buttons based on whether a spec is loaded or not
        const specLoaded = currentGraph !== null; // Or some other indicator that a spec is active

        if (submitSpecButton) submitSpecButton.disabled = false; // Always enable spec submission
        if (sendMessageButton) sendMessageButton.disabled = !specLoaded; // Enable if spec loaded
        if (runWorkflowButton) runWorkflowButton.disabled = !specLoaded; // Enable if spec loaded

        console.log("UI: Thinking stopped - ", message);
    }
    
    function connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Assuming your WebSocket endpoint is /ws/openai_spec_agent/{some_session_id}
        // For simplicity, using a fixed session_id or generating one client-side.
        // In a real app, this might be user-specific and managed.
        const sessionId = `client-${Date.now()}-${Math.random().toString(36).substring(2,7)}`;
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/openai_spec_agent/${sessionId}`;
        
        ws = new WebSocket(wsUrl);

        ws.onopen = function(event) {
            console.log("WebSocket connection established with session:", sessionId);
            addMessageToChat('status', "Connected to the agent.");
            stopThinking("Connected."); 
            // Initial button state
            if (sendMessageButton) sendMessageButton.disabled = true; // Disabled until spec loaded
            if (runWorkflowButton) runWorkflowButton.disabled = true; // Disabled until spec loaded
        };

        ws.onmessage = function(event) {
            try {
                const parsed_event_data = JSON.parse(event.data);
                console.log("WebSocket message received:", parsed_event_data);
                
                const messageType = parsed_event_data.type;
                const data = parsed_event_data.data || {};

                switch (messageType) {
                    case 'status_update': // General status updates, not for thinking
                        addMessageToChat('status', `Status: ${data.event} - ${data.message || ''}`);
                        break;
                    case 'intermediate_message':
                        addMessageToChat(data.source || 'assistant', data.message);
                        if (isThinking && data.message && thinkingIndicatorText) {
                            thinkingIndicatorText.textContent = data.message.length > 70 ? data.message.substring(0, 67) + "..." : data.message;
                        }
                        break;
                    case 'graph_update':
                        if (data.graph) {
                            currentGraph = data.graph; 
                            renderGraph(currentGraph);
                            addMessageToChat('status', "Workflow graph updated.");
                            if (sendMessageButton) sendMessageButton.disabled = false; // Enable interaction
                            if (runWorkflowButton) runWorkflowButton.disabled = false; // Enable run workflow
                        }
                        break;
                    case 'api_response':
                        displayApiResponse(data);
                        // This might be an intermediate step, so don't stop thinking unless it's the *final* part of a flow.
                        // For now, assuming `final_response` marks the true end.
                        break;
                    case 'final_response':
                        addMessageToChat('assistant', data.message);
                        if (data.graph) { // If final response also includes the graph
                            currentGraph = data.graph;
                            renderGraph(currentGraph);
                        }
                        if (data.result) { 
                            console.log("Final result data:", data.result);
                        }
                        stopThinking(data.message || "Task complete.");
                        if (currentGraph && sendMessageButton) sendMessageButton.disabled = false;
                        if (currentGraph && runWorkflowButton) runWorkflowButton.disabled = false;
                        break;
                    case 'error':
                        addMessageToChat('error', `Error: ${data.message}`);
                        stopThinking("An error occurred."); // Stop thinking on error
                        // Buttons remain enabled to allow user to try again or submit new spec
                        if (submitSpecButton) submitSpecButton.disabled = false;
                        if (sendMessageButton) sendMessageButton.disabled = (currentGraph === null);
                        if (runWorkflowButton) runWorkflowButton.disabled = (currentGraph === null);
                        break;
                    default:
                        addMessageToChat('unknown', `Received unhandled message type '${messageType}': ${event.data.substring(0,100)}...`);
                }
            } catch (e) {
                console.error("Error processing WebSocket message:", e, "Raw data:", event.data);
                addMessageToChat('error', "Received an unparseable message from server.");
                stopThinking("Error processing server message."); // Stop thinking if message parsing fails
            }
        };

        ws.onclose = function(event) {
            console.log("WebSocket connection closed.", event.reason, "Code:", event.code);
            addMessageToChat('status', "Connection closed. Attempting to reconnect in 5 seconds...");
            stopThinking("Connection lost."); 
            isThinking = false; // Ensure isThinking is false
            setTimeout(connectWebSocket, 5000); 
        };

        ws.onerror = function(error) {
            console.error("WebSocket error:", error);
            addMessageToChat('error', "WebSocket connection error. Check console for details.");
            // onclose will usually be called after onerror.
            // stopThinking() will be called by onclose.
        };
    }

    function addMessageToChat(source, message, type = 'text') {
        if (!chatMessages) return;
        const messageElement = document.createElement('div');
        messageElement.classList.add('mb-2', 'p-3', 'rounded-lg', 'max-w-2xl', 'break-words', 'text-sm', 'shadow-sm');

        let sourcePrefix = "";
        if (source === 'user') {
            messageElement.classList.add('bg-blue-500', 'text-white', 'self-end', 'ml-auto');
            sourcePrefix = '<strong class="font-semibold">You:</strong> ';
        } else if (source === 'assistant') {
            messageElement.classList.add('bg-gray-200', 'text-gray-800', 'self-start', 'mr-auto');
            sourcePrefix = '<strong class="font-semibold">Agent:</strong> ';
        } else if (source === 'status') {
            messageElement.classList.add('bg-yellow-100', 'text-yellow-800', 'self-center', 'text-xs', 'italic', 'text-center', 'py-1', 'px-2');
        } else if (source === 'error') {
            messageElement.classList.add('bg-red-100', 'text-red-700', 'self-start', 'mr-auto', 'font-semibold');
            sourcePrefix = '<strong class="font-semibold">Error:</strong> ';
        } else { 
            messageElement.classList.add('bg-gray-50', 'text-gray-500', 'self-center', 'text-xs', 'italic', 'text-center', 'py-1', 'px-2');
        }
        
        // Basic sanitization to prevent HTML injection if message comes from untrusted source
        const tempDiv = document.createElement('div');
        tempDiv.textContent = message;
        messageElement.innerHTML = sourcePrefix + tempDiv.innerHTML.replace(/\n/g, '<br>'); 
        
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight; 
    }

    function sendWebSocketMessage(type, payload) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            // Start thinking indicator *before* sending the message
            startThinking("Sending request..."); // Generic message, backend can update via intermediate_message
            
            const message = JSON.stringify({ type: type, ...payload });
            ws.send(message);
            console.log("Sent to WS:", message);
        } else {
            addMessageToChat('error', "WebSocket is not connected. Cannot send message.");
            console.error("WebSocket is not connected.");
            stopThinking("Connection error."); // Stop thinking if WS is not open
        }
    }

    if (submitSpecButton) {
        submitSpecButton.addEventListener('click', () => {
            let specData = specInput.value.trim();
            const url = specUrlInput.value.trim();
            const file = specFileInput.files[0];
            let source = 'text';
            let userMessage = "";

            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    specData = e.target.result;
                    source = 'file';
                    userMessage = `Submitted OpenAPI Spec from file: ${file.name}`;
                    addMessageToChat('user', userMessage);
                    sendWebSocketMessage('process_openapi_spec', { openapi_spec_string: specData, source: source, file_name: file.name });
                };
                reader.onerror = function(e) {
                    addMessageToChat('error', 'Error reading file.');
                    console.error("File reading error:", e);
                    stopThinking("File error."); // Stop thinking if file read fails
                };
                reader.readAsText(file);
            } else if (url) {
                specData = url; 
                source = 'url';
                userMessage = `Submitted OpenAPI Spec from URL: ${url}`;
                addMessageToChat('user', userMessage);
                sendWebSocketMessage('process_openapi_spec', { openapi_spec_url: specData, source: source });
            } else if (specData) {
                source = 'text';
                userMessage = `Submitted OpenAPI Spec (text input)`;
                addMessageToChat('user', userMessage);
                sendWebSocketMessage('process_openapi_spec', { openapi_spec_string: specData, source: source });
            } else {
                addMessageToChat('error', 'Please provide an OpenAPI spec (text, URL, or file).');
                return; // Don't send if no data
            }
        });
    }

    if (sendMessageButton) {
        sendMessageButton.addEventListener('click', () => {
            const messageText = userInput.value.trim();
            if (messageText) {
                addMessageToChat('user', messageText);
                // Pass current_graph if your backend needs it for context with user interactions
                sendWebSocketMessage('user_interaction', { text: messageText, current_graph: currentGraph ? currentGraph.elements : null });
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
            addMessageToChat('status', 'Chat and graph cleared.');
            if (sendMessageButton) sendMessageButton.disabled = true; // Disable until new spec loaded
            if (runWorkflowButton) runWorkflowButton.disabled = true; // Disable until new spec loaded
        });
    }
    
    if (closeApiResponseButton) {
        closeApiResponseButton.addEventListener('click', () => {
            if (apiResponseContainer) apiResponseContainer.classList.add('hidden');
        });
    }
    
    // Placeholder for runCurrentWorkflow if you have a button for it
    if (runWorkflowButton) { // Check if the button exists
        window.runCurrentWorkflow = function() { // Make it global for onclick or add event listener
            if (!currentGraph) {
                addMessageToChat('error', "No workflow graph is loaded to run.");
                return;
            }
            addMessageToChat('user', "Attempting to run the current workflow...");
            // This message type 'run_workflow' needs to be handled by your backend
            sendWebSocketMessage('run_workflow', { graph: currentGraph.elements, goal: currentGraph.goal || "Execute the current plan." });
        };
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
                const parsedBody = JSON.parse(bodyContent);
                bodyContent = JSON.stringify(parsedBody, null, 2);
            } catch (e) {
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
            try {
                unsafe = JSON.stringify(unsafe, null, 2);
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

    function renderGraph(graphData) {
        if (!graphContainer || !cytoscape) {
            console.error("Graph container or Cytoscape library not found.");
            if (!cytoscape) addMessageToChat('error', "Cytoscape library not loaded. Cannot render graph.");
            return;
        }
        graphContainer.innerHTML = ''; 

        try {
            // Ensure graphData.elements exists and is an array or object
            let elementsToRender = graphData.elements;
            if (!elementsToRender) {
                if (graphData.nodes && graphData.edges) { // Handle if elements is nested
                    elementsToRender = { nodes: graphData.nodes, edges: graphData.edges };
                } else {
                    console.warn("Graph data.elements is missing or in unexpected format. Attempting to use graphData directly.", graphData);
                    elementsToRender = graphData; // Fallback, might not work if format is wrong
                }
            }


            const cy = cytoscape({
                container: graphContainer,
                elements: elementsToRender,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'background-color': (ele) => {
                                const type = ele.data('type');
                                if (type === 'llm_call') return '#FF69B4'; // Pink
                                if (type === 'api_call') return '#1E90FF'; // DodgerBlue
                                if (type === 'user_input') return '#FFD700'; // Gold
                                if (type === 'knowledge_base') return '#32CD32'; // LimeGreen
                                if (type === 'start_node') return '#90EE90'; // LightGreen
                                if (type === 'end_node') return '#FA8072'; // Salmon
                                return '#666'; // Default
                            },
                            'label': 'data(label)',
                            'width': 'label',
                            'height': 'label',
                            'padding': '12px', 
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'shape': 'round-rectangle', 
                            'font-size': '10px', 
                            'color': '#fff', 
                            'text-outline-width': 1, 
                            'text-outline-color': '#333', 
                            'border-width': 2,
                            'border-color': '#4A5568' 
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': '#9CA3AF', 
                            'target-arrow-color': '#9CA3AF',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier', 
                            'label': 'data(label)', 
                            'font-size': '8px',
                            'color': '#4A5568',
                            'text-rotation': 'autorotate',
                            'text-margin-y': -10
                        }
                    },
                    { 
                        selector: 'node:selected',
                        style: {
                            'border-width': 3,
                            'border-color': '#F59E0B' 
                        }
                    }
                ],
                layout: {
                    name: 'dagre', 
                    rankDir: 'TB', 
                    spacingFactor: 1.3, // Increased spacing slightly
                    padding: 30 // Increased padding
                }
            });

            cy.nodes().forEach(node => {
                let content = `<strong>ID:</strong> ${node.id()}<br/><strong>Type:</strong> ${node.data('type') || 'N/A'}`;
                if (node.data('description')) {
                    content += `<br/><strong>Desc:</strong> ${escapeHtml(node.data('description'))}`;
                }
                if(node.data('status')) {
                    content += `<br/><strong>Status:</strong> ${node.data('status')}`;
                }
                
                if (typeof tippy === 'function') {
                    const tippyInstance = tippy(node.popperRef(), {
                        content: content,
                        trigger: 'manual', 
                        allowHTML: true,
                        placement: 'top',
                        arrow: true,
                        interactive: true, 
                        theme: 'light-border', 
                    });
                    node.scratch('tippy', tippyInstance); // Store for later use
                    
                    node.on('mouseover', (event) => event.target.scratch('tippy') && event.target.scratch('tippy').show());
                    node.on('mouseout', (event) => event.target.scratch('tippy') && event.target.scratch('tippy').hide());
                    node.on('tap', function(event) {
                        console.log('Clicked node:', event.target.data());
                        addMessageToChat('status', `Node clicked: ${event.target.data('label') || event.target.id()}`);
                    });

                } else if (!window.tippyWarningShown) {
                    console.warn("Tippy.js not loaded, tooltips will not be available for graph nodes.");
                    window.tippyWarningShown = true; // Show warning only once
                }
            });
            cy.fit(null, 30); 

        } catch (e) {
            console.error("Error rendering graph:", e, "Graph data used:", graphData);
            addMessageToChat('error', "Failed to render workflow graph. Check console.");
            if (graphContainer) graphContainer.innerHTML = '<p class="text-red-500 p-4">Error rendering graph. Please check the console for details.</p>';
        }
    }
    
    connectWebSocket();
});
