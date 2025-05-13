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

    // Modal Elements
    const confirmationModal = document.getElementById('confirmationModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalOperationId = document.getElementById('modalOperationId');
    const modalEffectiveNodeId = document.getElementById('modalEffectiveNodeId'); // Added for clarity
    const modalMethod = document.getElementById('modalMethod');
    const modalPath = document.getElementById('modalPath');
    const modalGraph2ThreadId = document.getElementById('modalGraph2ThreadId'); // To display G2 thread ID
    const modalPayload = document.getElementById('modalPayload');
    const modalConfirmButton = document.getElementById('modalConfirmButton');
    const modalCancelButton = document.getElementById('modalCancelButton');

    let ws;
    let currentGraphData = null;
    // Store details for the current confirmation including G2 thread ID and confirmation key
    let currentConfirmationContext = {
        graph2ThreadId: null,
        confirmationKey: null,
        operationId: null, // For logging/display
        effectiveNodeId: null // For logging/display
    };


    // --- Utility Functions ---
    function showThinking(show) {
        thinkingIndicator.style.display = show ? 'inline-block' : 'none';
        sendButton.disabled = show;
        userInput.disabled = show;
        runWorkflowButton.disabled = show; // Disable run workflow button while thinking
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            if (unsafe === null || unsafe === undefined) return '';
            try { 
                // For objects/arrays, pretty print with an indent of 2 for readability in <pre>
                return JSON.stringify(unsafe, null, 2); 
            } catch (e) { 
                return String(unsafe); 
            }
        }
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // --- Message Display Functions ---
    function addMessageToChat(sender, content, type, isWorkflowMessage = false) {
        const messageElement = document.createElement('div');
        let senderClass = 'agent-message'; // Default
        let senderName = sender;

        if (sender === 'You') {
            senderClass = 'user-message';
        } else if (isWorkflowMessage) {
            senderClass = 'workflow-message'; // Specific class for workflow messages
            senderName = `Workflow (${sender})`; // e.g., Workflow (tool_start)
        } else if (type === 'system' || sender === 'System') {
            senderClass = 'system-message';
            senderName = 'System';
        } else if (sender === 'Planner') {
            senderClass = 'planner-message';
        } else if (sender === 'Agent') {
             senderClass = 'agent-message';
        }


        messageElement.classList.add('message', senderClass);
        if (type === 'error') {
            messageElement.classList.add('error-message');
        }


        const senderElement = document.createElement('div');
        senderElement.classList.add('message-sender');
        senderElement.textContent = senderName;

        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');

        if (typeof content === 'string') {
            let formattedContent = content.replace(/```([\s\S]*?)```/g, (match, code) => {
                return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
            });
            formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedContent = formattedContent.replace(/\*(.*?)\*/g, '<em>$1</em>');
            contentElement.innerHTML = formattedContent.split('\n').map(line => `<p>${line}</p>`).join('');
        } else if (typeof content === 'object' && content !== null) {
            // Handle specific structured content for workflow messages
            if (isWorkflowMessage && content.node_name) { // Example for tool_start/tool_end
                 let detailsHtml = `<strong>Node:</strong> ${escapeHtml(content.node_name)}<br>`;
                 if (content.input_preview) detailsHtml += `Input Preview: <pre>${escapeHtml(content.input_preview)}</pre>`;
                 if (content.status_code) detailsHtml += `Status: ${escapeHtml(content.status_code)} `;
                 if (content.execution_time) detailsHtml += `(Took: ${escapeHtml(content.execution_time)}s)<br>`;
                 if (content.response_preview) detailsHtml += `Response Preview: <pre>${escapeHtml(content.response_preview)}</pre>`;
                 if (content.error) detailsHtml += `<strong style="color: red;">Error:</strong> <pre>${escapeHtml(content.error)}</pre>`;
                 if (content.message) detailsHtml += `<p>${escapeHtml(content.message)}</p>`;
                 if (content.final_state) detailsHtml += `Final State Keys: <pre>${escapeHtml(Object.keys(content.final_state || {}).join(', '))}</pre>`;

                 contentElement.innerHTML = detailsHtml;
            } else if (content.message) { // General message object
                contentElement.innerHTML = `<p>${escapeHtml(content.message)}</p>`;
                 if (content.details) { // If there are more details
                    contentElement.innerHTML += `<pre>${escapeHtml(content.details)}</pre>`;
                 }
            } else { // Fallback for other objects
                contentElement.innerHTML = `<pre>${escapeHtml(content)}</pre>`;
            }
        } else {
             contentElement.innerHTML = `<p>${escapeHtml(String(content))}</p>`;
        }
        
        messageElement.appendChild(senderElement);
        messageElement.appendChild(contentElement);
        chatMessagesDiv.appendChild(messageElement);
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
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
            runWorkflowButton.disabled = true; // Initially disabled until a graph is loaded
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("WS RX:", data); 
                showThinking(false); 

                let sender = 'Agent'; // Default sender
                let messageContent = data.content;
                let messageType = data.type; // e.g. 'final', 'intermediate', 'error', 'tool_start'
                let isWorkflowMsg = false;

                if (data.source === 'graph1_planning') {
                    sender = 'Planner';
                    if (data.type === "graph_update") {
                        currentGraphData = data.content;
                        graphJsonViewPre.textContent = JSON.stringify(currentGraphData, null, 2);
                        addMessageToChat('System', 'Execution graph has been updated.', 'system');
                        if (graphDagViewContent.classList.contains('active')) {
                            renderMermaidGraphUI(currentGraphData);
                        }
                        runWorkflowButton.disabled = !currentGraphData; // Enable if graph exists
                        return; // Handled graph update
                    }
                    messageContent = data.content.message || data.content;
                } else if (data.source === 'graph2_execution') {
                    sender = data.type; // Use event type like 'tool_start', 'tool_end' as sender for workflow
                    isWorkflowMsg = true;
                    // Specific handling for workflow completion/failure to re-enable button
                    if (data.type === "execution_completed" || data.type === "execution_failed" || data.type === "workflow_timeout") {
                        runWorkflowButton.disabled = !currentGraphData; // Re-enable based on graph presence
                    }
                     if (data.type === "human_intervention_required") {
                        // Pass the full data.content (which is {node_name, details_for_ui})
                        // and data.graph2_thread_id to showConfirmationModal
                        showConfirmationModal(data.content.details_for_ui, data.graph2_thread_id);
                        // Add a message to chat indicating pause
                        addMessageToChat(
                            `Workflow (${data.content.node_name})`,
                            `Paused. Requires confirmation for operation: ${data.content.details_for_ui.operationId}. Please use the modal.`,
                            'interrupt_confirmation_required',
                            true
                        );
                        return; // Modal handles further interaction
                    }
                } else if (data.source === 'system' || data.source === 'system_error' || data.source === 'system_warning') {
                    sender = 'System';
                    messageContent = data.content.error || data.content.message || data.content;
                }


                addMessageToChat(sender, messageContent, messageType, isWorkflowMsg);

            } catch (error) {
                console.error("Error processing WebSocket message:", error, "Raw data:", event.data);
                addMessageToChat('System', 'Error processing message from server. Check console.', 'error');
                showThinking(false);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            addMessageToChat('System', 'WebSocket connection error. Check console.', 'error');
            showThinking(false);
            runWorkflowButton.disabled = true;
        };

        ws.onclose = (event) => {
            addMessageToChat('System', `WebSocket disconnected. Code: ${event.code}. Attempting to reconnect in 5 seconds...`, 'system');
            showThinking(false);
            runWorkflowButton.disabled = true;
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
        userInput.style.height = 'auto'; 
        showThinking(true);
    };

    window.runCurrentWorkflow = () => {
        if (!currentGraphData) {
            addMessageToChat('System', 'No workflow graph loaded to run.', 'error');
            return;
        }
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessageToChat('System', 'Not connected. Cannot run workflow.', 'error');
            return;
        }
        // The backend router will interpret "run workflow" or similar,
        // and then Graph 1's "setup_workflow_execution" node will set the stage.
        // The actual execution start is triggered by main.py observing BotState.
        const command = "run workflow"; 
        addMessageToChat('You', command, 'user');
        ws.send(command);
        addMessageToChat('System', "Requesting to run the current workflow...", 'system', true);
        showThinking(true);
        runWorkflowButton.disabled = true; // Disable while attempting to run
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
            graphJsonViewContent.style.display = 'flex'; // Use flex for consistency
            graphJsonViewContent.classList.add('active');
            jsonTabButton.classList.add('active');
        } else if (tabName === 'dag') {
            graphDagViewContent.style.display = 'flex'; 
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
        def += "    classDef startEnd fill:#6c757d,stroke:#5a6268,stroke-width:2px,color:white,font-weight:bold,rx:8,ry:8,padding:10px;\n";
        def += "    classDef apiNode fill:#007bff,stroke:#0056b3,stroke-width:2px,color:white,rx:8,ry:8,padding:10px;\n";
        def += "    classDef confirmedNode fill:#28a745,stroke:#1e7e34,stroke-width:2px,color:white,rx:8,ry:8,padding:10px;\n"; // Green for confirmed
        def += "    classDef skippedNode fill:#ffc107,stroke:#cca408,stroke-width:2px,color:black,rx:8,ry:8,padding:10px;\n"; // Yellow for skipped
        def += "    classDef errorNode fill:#dc3545,stroke:#b02a37,stroke-width:2px,color:white,rx:8,ry:8,padding:10px;\n"; // Red for error


        const sanitizeId = (id) => String(id).replace(/[^a-zA-Z0-9_]/g, '_');

        graph.nodes.forEach(node => {
            const id = sanitizeId(node.effective_id || node.operationId);
            let labelText = node.summary || node.operationId;
            if (node.summary && node.summary !== node.operationId && node.operationId !== "START_NODE" && node.operationId !== "END_NODE") {
                labelText = `${node.summary}<br/><small>(${node.operationId} / ${node.effective_id})</small>`;
            } else {
                labelText = `<b>${node.operationId}</b><br/><small>(${node.effective_id})</small>`;
            }
            def += `    ${id}("${labelText}");\n`;
            if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                def += `    class ${id} startEnd;\n`;
            } else {
                def += `    class ${id} apiNode;\n`; // Default class
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
            mermaidDagContainer.innerHTML = "<p>Graph not yet available or cannot be rendered.</p>";
            return;
        }
        if (typeof mermaid === 'undefined') {
            mermaidDagContainer.innerHTML = "<p>Mermaid.js library not loaded.</p>";
            return;
        }
        try {
            const definition = generateMermaidDefinition(graphData);
            mermaidDagContainer.innerHTML = ""; // Clear previous
            const { svg } = await mermaid.render('mermaidGeneratedSvg', definition);
            mermaidDagContainer.innerHTML = svg;
        } catch (error) {
            console.error("Mermaid rendering error:", error, "\nDefinition:", generateMermaidDefinition(graphData));
            mermaidDagContainer.textContent = "Error rendering DAG. Check console.";
        }
    }
    
    // --- Confirmation Modal Functions ---
    function showConfirmationModal(details, graph2ThreadId) { // details is details_for_ui from backend
        if (!details) {
            console.error("No details provided for confirmation modal.");
            addMessageToChat('System', 'Error: Missing details for confirmation modal.', 'error', true);
            return;
        }
        
        currentConfirmationContext.graph2ThreadId = graph2ThreadId;
        currentConfirmationContext.confirmationKey = details.confirmation_key;
        currentConfirmationContext.operationId = details.operationId;
        currentConfirmationContext.effectiveNodeId = details.effective_node_id;


        modalTitle.textContent = details.prompt || `Confirm API Call: ${details.operationId}`;
        modalOperationId.textContent = details.operationId || 'N/A';
        modalEffectiveNodeId.textContent = details.effective_node_id || 'N/A';
        modalMethod.textContent = details.method || 'N/A';
        modalPath.textContent = details.path || 'N/A'; 
        modalGraph2ThreadId.textContent = graph2ThreadId || 'N/A';


        let payloadToDisplay = "";
        if (details.payload_to_confirm !== undefined && details.payload_to_confirm !== null) {
            try {
                payloadToDisplay = JSON.stringify(details.payload_to_confirm, null, 2);
            } catch (e) {
                payloadToDisplay = "Error: Could not format payload.";
                console.error("Error stringifying payload_to_confirm:", details.payload_to_confirm, e);
            }
        }
        modalPayload.value = payloadToDisplay;
        confirmationModal.style.display = 'flex';
    }

    function hideConfirmationModal() {
        confirmationModal.style.display = 'none';
        currentConfirmationContext.graph2ThreadId = null;
        currentConfirmationContext.confirmationKey = null;
        currentConfirmationContext.operationId = null;
        currentConfirmationContext.effectiveNodeId = null;
        modalPayload.value = ''; // Clear payload
    }

    modalConfirmButton.onclick = () => {
        if (!currentConfirmationContext.graph2ThreadId || !currentConfirmationContext.confirmationKey) {
            addMessageToChat('System', 'Error: Missing context for confirmation (Graph2 Thread ID or Confirmation Key). Cannot send resume command.', 'error', true);
            hideConfirmationModal();
            return;
        }

        let parsedPayload;
        try {
            if (modalPayload.value.trim() === "") {
                // If payload is empty, backend might expect null or original.
                // For now, send null if empty, or an empty object if that's more appropriate.
                // The backend's _apply_confirmed_data_to_request should handle this.
                parsedPayload = {}; // Or null, depending on backend expectation for "empty"
            } else {
                parsedPayload = JSON.parse(modalPayload.value);
            }
        } catch (e) {
            alert('Invalid JSON in payload textarea: ' + e.message);
            addMessageToChat('System', 'Payload in modal is not valid JSON. Please correct it.', 'error', true);
            return;
        }

        const resumeData = {
            confirmation_key: currentConfirmationContext.confirmationKey,
            decision: true,
            modified_payload: parsedPayload,
            // Include other details from currentConfirmationContext if needed by backend
            operationId: currentConfirmationContext.operationId, 
            effectiveNodeId: currentConfirmationContext.effectiveNodeId
        };
        
        // Correctly format the resume_exec command
        const wsMessage = `resume_exec ${currentConfirmationContext.graph2ThreadId} ${JSON.stringify(resumeData)}`;
        
        addMessageToChat(
            `You (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, // Shortened G2 ID
            `Confirming: ${currentConfirmationContext.operationId || 'action'} with payload.`, 
            'user'
        );
        ws.send(wsMessage);
        addMessageToChat(
            `System (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, 
            `Confirmation sent for ${currentConfirmationContext.effectiveNodeId}. Waiting for workflow to resume...`, 
            'system', 
            true
        );
        hideConfirmationModal();
        showThinking(true); 
    };

    modalCancelButton.onclick = () => {
        if (!currentConfirmationContext.graph2ThreadId || !currentConfirmationContext.confirmationKey) {
            addMessageToChat('System', 'Error: Missing context for cancellation. Cannot send resume command.', 'error', true);
            hideConfirmationModal();
            return;
        }
        const resumeData = {
            confirmation_key: currentConfirmationContext.confirmationKey,
            decision: false, // User cancelled
            operationId: currentConfirmationContext.operationId,
            effectiveNodeId: currentConfirmationContext.effectiveNodeId
        };
        const wsMessage = `resume_exec ${currentConfirmationContext.graph2ThreadId} ${JSON.stringify(resumeData)}`;
        
        addMessageToChat(
            `You (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, 
            `Cancelling confirmation for: ${currentConfirmationContext.operationId || 'action'}.`, 
            'user'
        );
        ws.send(wsMessage);
         addMessageToChat(
            `System (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, 
            `Cancellation sent for ${currentConfirmationContext.effectiveNodeId}.`, 
            'system', 
            true
        );
        hideConfirmationModal();
        // No thinking indicator for cancel, as it's a quick state update usually.
        // Backend will send further updates if workflow terminates or continues differently.
    };


    // --- Initial Setup ---
    userInput.addEventListener('input', () => { 
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    showGraphTab('json'); 
    connectWebSocket(); 
});
