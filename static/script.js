// OpenAPI Agent script.js

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatMessagesDiv = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const thinkingIndicator = document.getElementById('thinkingIndicator');
    const runWorkflowButton = document.getElementById('runWorkflowButton');

    const graphJsonViewPre = document.getElementById('graphJsonViewPre');
    const mermaidDagContainer = document.getElementById('mermaidDagDiagram');
    const jsonTabButton = document.getElementById('jsonTabButton');
    const dagTabButton = document.getElementById('dagTabButton');
    const graphJsonViewContent = document.getElementById('graphJsonViewContent');
    const graphDagViewContent = document.getElementById('graphDagViewContent');

    const workflowLogMessagesDiv = document.getElementById('workflowLogMessages');

    // Modal Elements
    const confirmationModal = document.getElementById('confirmationModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalOperationId = document.getElementById('modalOperationId');
    const modalMethod = document.getElementById('modalMethod');
    const modalPath = document.getElementById('modalPath');
    const modalPayload = document.getElementById('modalPayload');
    const modalConfirmButton = document.getElementById('modalConfirmButton');
    const modalCancelButton = document.getElementById('modalCancelButton');

    let ws;
    let currentGraphData = null;
    let currentInterruptionData = null; // Stores data for the active confirmation

    // --- Utility Functions ---
    function showThinking(show) {
        thinkingIndicator.style.display = show ? 'inline-block' : 'none';
        sendButton.disabled = show;
        userInput.disabled = show;
        runWorkflowButton.disabled = show;
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            if (typeof unsafe === 'object' && unsafe !== null) {
                try { return JSON.stringify(unsafe, null, 2); } // Pretty print objects
                catch (e) { return String(unsafe); }
            }
            return String(unsafe);
        }
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // --- Message Display Functions ---
    function addMessageToChat(sender, content, type) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${type}-message`); // e.g., user-message, agent-message

        const senderElement = document.createElement('div');
        senderElement.classList.add('message-sender');
        senderElement.textContent = sender;

        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');

        // Basic Markdown-like formatting (simple version)
        if (typeof content === 'string') {
            // Handle code blocks ```text```
            let formattedContent = content.replace(/```([\s\S]*?)```/g, (match, code) => {
                return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
            });
            // Handle bold **text** and italic *text*
            formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedContent = formattedContent.replace(/\*(.*?)\*/g, '<em>$1</em>');
            // Handle newlines
            contentElement.innerHTML = formattedContent.split('\n').map(line => `<p>${line}</p>`).join('');
        } else { // If content is an object, pretty-print it
            contentElement.innerHTML = `<pre>${escapeHtml(content)}</pre>`;
        }
        
        messageElement.appendChild(senderElement);
        messageElement.appendChild(contentElement);
        chatMessagesDiv.appendChild(messageElement);
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
    }

    function addWorkflowLog(logEntry, type = 'info') {
        const logElement = document.createElement('div');
        logElement.classList.add('workflow-log-entry', `workflow-${type}`);
        const timestamp = `<span class="log-timestamp">${new Date().toLocaleTimeString()}</span>`;
        if (typeof logEntry === 'object') {
            logElement.innerHTML = `${timestamp} <pre>${escapeHtml(JSON.stringify(logEntry, null, 2))}</pre>`;
        } else {
            logElement.innerHTML = `${timestamp} ${escapeHtml(logEntry)}`;
        }
        workflowLogMessagesDiv.appendChild(logElement);
        workflowLogMessagesDiv.scrollTop = workflowLogMessagesDiv.scrollHeight;
    }

    // --- WebSocket Functions ---
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/openapi_agent`;
        
        addMessageToChat('System', `Attempting to connect to ${wsUrl}...`, 'system');
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            addMessageToChat('System', 'Successfully connected to the agent.', 'system');
            showThinking(false);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("WS RX:", data); // Log all incoming messages for debugging

                const sender = data.source === 'graph1_planning' ? 'Planner' :
                               data.source === 'graph2_execution' ? 'Workflow' :
                               'Agent'; // Default to Agent if source is system or unclear

                showThinking(false); // Generally stop thinking unless a new processing starts

                switch (data.type) {
                    case 'status':
                        if (data.content && typeof data.content === 'string' && data.content.toLowerCase().includes("processing")) {
                            showThinking(true);
                        }
                        addMessageToChat(sender, data.content, data.type);
                        break;
                    case 'info':
                    case 'final':
                    case 'error':
                    case 'intermediate':
                        addMessageToChat(sender, data.content, data.type);
                        break;
                    case 'graph_update':
                        currentGraphData = data.content;
                        graphJsonViewPre.textContent = JSON.stringify(currentGraphData, null, 2);
                        addMessageToChat('System', 'Execution graph has been updated.', 'system');
                        if (graphDagViewContent.classList.contains('active')) {
                            renderMermaidGraphUI(currentGraphData);
                        }
                        break;
                    case 'human_intervention_required':
                        addWorkflowLog(data.content, 'interrupt_confirmation_required');
                        // The content here is expected to be data.content.details_for_ui
                        // which matches interrupt_data_for_ui from backend
                        showConfirmationModal(data.content.details_for_ui); 
                        break;
                    // Specific workflow log types can be added here if backend sends them
                    // For now, most workflow messages might come as 'info', 'error' with source 'graph2_execution'
                    default:
                        if (data.source === 'graph2_execution') {
                            addWorkflowLog(data.content, data.type); // Log with original type
                            // Optionally, also add a chat message for important workflow events
                            if (data.type === 'execution_completed' || data.type === 'execution_failed' || data.type === 'workflow_timeout') {
                                addMessageToChat('Workflow', data.content.message || JSON.stringify(data.content), data.type);
                            }
                        } else {
                            addMessageToChat(sender, data.content, data.type); // Fallback
                        }
                }
            } catch (error) {
                console.error("Error processing WebSocket message:", error, "Raw data:", event.data);
                addMessageToChat('System', 'Error processing message from server.', 'error');
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            addMessageToChat('System', 'WebSocket connection error. Check console.', 'error');
            showThinking(false);
        };

        ws.onclose = (event) => {
            addMessageToChat('System', `WebSocket disconnected. Code: ${event.code}. Attempting to reconnect in 5 seconds...`, 'system');
            showThinking(false);
            setTimeout(connectWebSocket, 5000);
        };
    }

    window.sendMessage = () => {
        const messageText = userInput.value.trim();
        if (!messageText) return;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessageToChat('System', 'Not connected. Please wait or try refreshing.', 'error');
            return;
        }
        addMessageToChat('You', messageText, 'user');
        ws.send(messageText);
        userInput.value = '';
        userInput.style.height = 'auto'; // Reset height
        showThinking(true);
    };

    window.runCurrentWorkflow = () => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessageToChat('System', 'Not connected. Cannot run workflow.', 'error');
            return;
        }
        const command = "run current workflow"; // Backend router should understand this
        addMessageToChat('You', command, 'user');
        ws.send(command);
        addWorkflowLog("User initiated workflow run.", 'info');
        showThinking(true);
    };

    // --- Graph Tab and Mermaid Rendering ---
    window.showGraphTab = (tabName) => {
        graphJsonViewContent.style.display = 'none';
        graphDagViewContent.style.display = 'none';
        graphJsonViewContent.classList.remove('active');
        graphDagViewContent.classList.remove('active');
        jsonTabButton.classList.remove('active');
        dagTabButton.classList.remove('active');

        if (tabName === 'json') {
            graphJsonViewContent.style.display = 'block';
            graphJsonViewContent.classList.add('active');
            jsonTabButton.classList.add('active');
        } else if (tabName === 'dag') {
            graphDagViewContent.style.display = 'block'; // Display before rendering
            graphDagViewContent.classList.add('active');
            dagTabButton.classList.add('active');
            renderMermaidGraphUI(currentGraphData);
        }
    };

    function generateMermaidDefinition(graph) {
        if (!graph || !graph.nodes || !graph.edges) {
            return "graph TD\n    empty[\"No graph data or invalid format.\"];";
        }
        let def = "graph TD;\n";
        def += "    classDef startEnd fill:#6c757d,stroke:#5a6268,stroke-width:2px,color:white,font-weight:bold,rx:8,ry:8;\n";
        def += "    classDef apiNode fill:#007bff,stroke:#0056b3,stroke-width:2px,color:white,rx:8,ry:8;\n";
        // Add more classDefs for running, success, error if you want to color nodes during execution

        const sanitizeId = (id) => id.replace(/[^a-zA-Z0-9_]/g, '_');

        graph.nodes.forEach(node => {
            const id = sanitizeId(node.effective_id || node.operationId);
            let label = escapeHtml(node.summary || node.operationId);
            if (node.summary && node.summary !== node.operationId && node.operationId !== "START_NODE" && node.operationId !== "END_NODE") {
                label = `${escapeHtml(node.summary)}<br/><small>(${escapeHtml(node.operationId)})</small>`;
            } else {
                label = `<b>${escapeHtml(node.operationId)}</b>`;
            }
            def += `    ${id}("${label}");\n`;
            if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                def += `    class ${id} startEnd;\n`;
            } else {
                def += `    class ${id} apiNode;\n`;
            }
        });

        graph.edges.forEach(edge => {
            const from = sanitizeId(edge.from_node);
            const to = sanitizeId(edge.to_node);
            const label = edge.description ? `|"${escapeHtml(edge.description.substring(0, 50))}"|` : "";
            def += `    ${from} -->${label} ${to};\n`;
        });
        return def;
    }

    async function renderMermaidGraphUI(graphData) {
        if (!graphData) {
            mermaidDagContainer.innerHTML = "Graph not yet available or cannot be rendered.";
            return;
        }
        if (typeof mermaid === 'undefined') {
            mermaidDagContainer.innerHTML = "Mermaid.js library not loaded.";
            return;
        }
        try {
            const definition = generateMermaidDefinition(graphData);
            mermaidDagContainer.innerHTML = ""; // Clear previous
            const { svg } = await mermaid.render('mermaidGeneratedSvg', definition);
            mermaidDagContainer.innerHTML = svg;
        } catch (error) {
            console.error("Mermaid rendering error:", error);
            mermaidDagContainer.textContent = "Error rendering DAG. Check console.";
        }
    }
    
    // --- Confirmation Modal Functions ---
    function showConfirmationModal(data) {
        if (!data) {
            console.error("No data provided for confirmation modal.");
            return;
        }
        currentInterruptionData = data; // data is interrupt_data_for_ui from backend

        modalTitle.textContent = data.prompt || `Confirm API Call: ${data.operationId}`;
        modalOperationId.textContent = data.operationId || 'N/A';
        modalMethod.textContent = data.method || 'N/A';
        modalPath.textContent = data.path || 'N/A'; // Backend sends 'path'

        let payloadToDisplay = "";
        if (data.payload_to_confirm !== undefined && data.payload_to_confirm !== null) {
            try {
                payloadToDisplay = JSON.stringify(data.payload_to_confirm, null, 2);
            } catch (e) {
                payloadToDisplay = "Error: Could not format payload.";
                console.error("Error stringifying payload_to_confirm:", data.payload_to_confirm, e);
            }
        }
        modalPayload.value = payloadToDisplay;
        confirmationModal.style.display = 'flex';
    }

    window.hideConfirmationModal = () => {
        confirmationModal.style.display = 'none';
        currentInterruptionData = null; // Clear data when modal is hidden
    };

    modalConfirmButton.onclick = () => {
        if (!currentInterruptionData || !currentInterruptionData.confirmation_key) {
            addMessageToChat('System', 'Error: Missing data for confirmation.', 'error');
            hideConfirmationModal();
            return;
        }

        let modifiedPayload;
        try {
            if (modalPayload.value.trim() === "") {
                modifiedPayload = null; // Or currentInterruptionData.payload_to_confirm if no change means original
            } else {
                modifiedPayload = JSON.parse(modalPayload.value);
            }
        } catch (e) {
            alert('Invalid JSON in payload textarea.');
            addMessageToChat('System', 'Payload is not valid JSON. Please correct it.', 'error');
            return;
        }

        const resumePayload = {
            confirmation_key: currentInterruptionData.confirmation_key,
            decision: true,
            modified_payload: modifiedPayload,
            // Include other fields if your modal allows editing them, e.g.:
            // modified_query_params: currentInterruptionData.query_params_to_confirm,
            // modified_headers: currentInterruptionData.headers_to_confirm,
        };

        const wsMessage = `resume_exec ${JSON.stringify(resumePayload)}`;
        addMessageToChat('You', `Confirming: ${currentInterruptionData.operationId}`, 'user');
        ws.send(wsMessage);
        addWorkflowLog(`User confirmed operation: ${currentInterruptionData.operationId}`, 'info');
        hideConfirmationModal();
        showThinking(true); // Show thinking while backend processes resume
    };

    // --- Initial Setup ---
    userInput.addEventListener('input', () => { // Auto-resize textarea
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    showGraphTab('json'); // Default to JSON tab
    connectWebSocket(); // Start WebSocket connection
});
