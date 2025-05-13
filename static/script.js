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
    const modalEffectiveNodeId = document.getElementById('modalEffectiveNodeId'); 
    const modalMethod = document.getElementById('modalMethod');
    const modalPath = document.getElementById('modalPath');
    const modalGraph2ThreadId = document.getElementById('modalGraph2ThreadId'); 
    const modalPayload = document.getElementById('modalPayload');
    const modalConfirmButton = document.getElementById('modalConfirmButton');
    const modalCancelButton = document.getElementById('modalCancelButton');

    let ws;
    let currentGraphData = null;
    let currentConfirmationContext = {
        graph2ThreadId: null,
        confirmationKey: null,
        operationId: null, 
        effectiveNodeId: null 
    };

    function showThinking(show) {
        thinkingIndicator.style.display = show ? 'inline-block' : 'none';
        sendButton.disabled = show;
        userInput.disabled = show;
        runWorkflowButton.disabled = show; 
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            if (unsafe === null || unsafe === undefined) return '';
            try { 
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

    function addMessageToChat(sender, content, type, sourceGraph = "graph1_planning") {
        const messageElement = document.createElement('div');
        let senderClass = 'agent-message'; 
        let senderName = sender;

        if (sender === 'You') {
            senderClass = 'user-message';
        } else if (sourceGraph === "graph2_execution") {
            senderClass = 'workflow-message'; 
            senderName = `Workflow (${type})`; 
        } else if (sourceGraph === "system" || sender === "System") {
            senderClass = 'system-message';
            senderName = 'System';
        } else if (sender === 'Planner') { 
            senderClass = 'planner-message';
        }
        
        messageElement.classList.add('message', senderClass);
        if (type === 'error' || (typeof content === 'object' && content && content.error)) {
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
            if (sourceGraph === "graph2_execution") {
                 let detailsHtml = "";
                 if (content.node_name) detailsHtml += `<strong>Node:</strong> ${escapeHtml(content.node_name)}<br>`;
                 if (content.input_preview) detailsHtml += `Input Preview: <pre>${escapeHtml(content.input_preview)}</pre>`;
                 if (content.status_code) detailsHtml += `Status: ${escapeHtml(content.status_code)} `;
                 if (content.execution_time) detailsHtml += `(Took: ${escapeHtml(content.execution_time)}s)<br>`;
                 if (content.response_preview) detailsHtml += `Response Preview: <pre>${escapeHtml(content.response_preview)}</pre>`;
                 if (content.raw_output_from_node && type === "tool_end") { 
                    detailsHtml += `Raw Output (State Update): <pre>${escapeHtml(content.raw_output_from_node)}</pre>`;
                 }
                 if (content.error) detailsHtml += `<strong style="color: var(--error-color);">Error:</strong> <pre>${escapeHtml(content.error)}</pre>`;
                 
                 if (content.message && (type === "execution_completed" || type === "execution_failed" || type === "workflow_timeout" || type === "info" || type === "error" )) {
                    detailsHtml += `<p>${escapeHtml(content.message)}</p>`;
                 }
                 if (content.final_state && (type === "execution_completed" || type === "execution_failed")) {
                    detailsHtml += `Final State (details): <pre>${escapeHtml(content.final_state)}</pre>`;
                 }
                 contentElement.innerHTML = detailsHtml || `<pre>${escapeHtml(content)}</pre>`; 
            } else if (content.message) { 
                contentElement.innerHTML = `<p>${escapeHtml(content.message)}</p>`;
                 if (content.details) { 
                    contentElement.innerHTML += `<pre>${escapeHtml(content.details)}</pre>`;
                 }
            } else { 
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

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/openapi_agent`;
        
        addMessageToChat('System', `Attempting to connect to ${wsUrl}...`, 'system', 'system');
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            addMessageToChat('System', 'Successfully connected to the agent.', 'system', 'system');
            showThinking(false);
            runWorkflowButton.disabled = true; 
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("WS RX:", data); 
                showThinking(false); 

                let sender = 'Agent'; 
                let messageContent = data.content;
                let messageType = data.type; 
                let sourceGraph = data.source || "unknown_source";

                if (sourceGraph === 'graph1_planning') {
                    sender = 'Planner';
                    if (data.type === "graph_update") {
                        currentGraphData = data.content;
                        graphJsonViewPre.textContent = JSON.stringify(currentGraphData, null, 2);
                        addMessageToChat('System', 'Execution graph has been updated.', 'system', 'system');
                        if (graphDagViewContent.classList.contains('active')) {
                            renderMermaidGraphUI(currentGraphData);
                        }
                        runWorkflowButton.disabled = !currentGraphData; 
                        return; 
                    }
                    messageContent = data.content.message || data.content; 
                } else if (sourceGraph === 'graph2_execution') {
                    if (data.type === "execution_completed" || data.type === "execution_failed" || data.type === "workflow_timeout") {
                        runWorkflowButton.disabled = !currentGraphData; 
                    }
                     if (data.type === "human_intervention_required") {
                        showConfirmationModal(data.content.details_for_ui, data.graph2_thread_id);
                        addMessageToChat(
                            `Workflow (${data.content.node_name || 'N/A'})`, 
                            `Paused. Requires confirmation for operation: ${data.content.details_for_ui.operationId}. Please use the modal.`,
                            'human_intervention_required', 
                            sourceGraph 
                        );
                        return; 
                    }
                } else if (sourceGraph === 'system' || sourceGraph === 'system_error' || sourceGraph === 'system_warning') {
                    sender = 'System';
                    messageContent = data.content.error || data.content.message || data.content;
                }
                addMessageToChat(sender, messageContent, messageType, sourceGraph);

            } catch (error) {
                console.error("Error processing WebSocket message:", error, "Raw data:", event.data);
                addMessageToChat('System', 'Error processing message from server. Check console.', 'error', 'system');
                showThinking(false);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            addMessageToChat('System', 'WebSocket connection error. Check console.', 'error', 'system');
            showThinking(false);
            runWorkflowButton.disabled = true;
        };

        ws.onclose = (event) => {
            addMessageToChat('System', `WebSocket disconnected. Code: ${event.code}. Attempting to reconnect in 5 seconds...`, 'system', 'system');
            showThinking(false);
            runWorkflowButton.disabled = true;
            setTimeout(connectWebSocket, 5000);
        };
    }

    window.sendMessage = () => {
        const messageText = userInput.value.trim();
        if (!messageText) return;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessageToChat('System', 'Not connected. Please wait or try refreshing.', 'error', 'system');
            return;
        }
        addMessageToChat('You', messageText, 'user', 'user_input'); 
        ws.send(messageText);
        userInput.value = '';
        userInput.style.height = 'auto'; 
        showThinking(true);
    };

    window.runCurrentWorkflow = () => {
        if (!currentGraphData) {
            addMessageToChat('System', 'No workflow graph loaded to run.', 'error', 'system');
            return;
        }
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessageToChat('System', 'Not connected. Cannot run workflow.', 'error', 'system');
            return;
        }
        const command = "run workflow"; 
        addMessageToChat('You', command, 'user', 'user_input');
        ws.send(command);
        addMessageToChat('System', "Requesting to run the current workflow...", 'status', 'system');
        showThinking(true);
        runWorkflowButton.disabled = true; 
    };

    window.showGraphTab = (tabName) => {
        graphJsonViewContent.style.display = 'none';
        graphDagViewContent.style.display = 'none';
        graphJsonViewContent.classList.remove('active');
        graphDagViewContent.classList.remove('active');
        jsonTabButton.classList.remove('active');
        dagTabButton.classList.remove('active');

        if (tabName === 'json') {
            graphJsonViewContent.style.display = 'flex'; 
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
        // More distinct styling for nodes
        def += "    classDef startEnd fill:#607d8b,stroke:#455a64,stroke-width:2px,color:white,font-weight:bold,rx:8,ry:8,padding:10px 15px;\n"; // Darker Grey
        def += "    classDef apiNode fill:var(--primary-color),stroke:var(--primary-hover-color),stroke-width:2px,color:white,rx:8,ry:8,padding:10px 15px;\n";
        def += "    classDef confirmedNode fill:var(--success-color),stroke:#1e7e34,stroke-width:2px,color:white,rx:8,ry:8,padding:10px 15px;\n"; 
        def += "    classDef skippedNode fill:var(--warning-color),stroke:#b8860b,stroke-width:2px,color:#333,rx:8,ry:8,padding:10px 15px;\n"; 
        def += "    classDef errorNode fill:var(--error-color),stroke:#a71d2a,stroke-width:2px,color:white,rx:8,ry:8,padding:10px 15px;\n"; 
        def += "    classDef interruptNode fill:var(--accent-color),stroke:#0f6c7a,stroke-width:2px,color:white,rx:8,ry:8,padding:10px 15px;\n"; // For nodes requiring confirmation


        const sanitizeId = (id) => String(id).replace(/[^a-zA-Z0-9_]/g, '_');

        graph.nodes.forEach(node => {
            const id = sanitizeId(node.effective_id || node.operationId);
            let labelText = node.summary || node.operationId;
            if (node.summary && node.summary !== node.operationId && node.operationId !== "START_NODE" && node.operationId !== "END_NODE") {
                labelText = `${escapeHtml(node.summary)}<br/><small>(${escapeHtml(node.operationId)} / ${escapeHtml(node.effective_id)})</small>`;
            } else {
                labelText = `<b>${escapeHtml(node.operationId)}</b><br/><small>(${escapeHtml(node.effective_id)})</small>`;
            }
            def += `    ${id}("${labelText}");\n`;
            if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                def += `    class ${id} startEnd;\n`;
            } else if (node.requires_confirmation) {
                 def += `    class ${id} interruptNode;\n`;
            }
            else {
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
            mermaidDagContainer.innerHTML = "<p style='text-align:center; color: var(--text-muted-color); padding: 20px;'>Graph not yet available. Generate a plan to see the visualization.</p>";
            return;
        }
        if (typeof mermaid === 'undefined') {
            mermaidDagContainer.innerHTML = "<p style='text-align:center; color: var(--error-color);'>Mermaid.js library not loaded. Cannot render graph.</p>";
            return;
        }
        mermaidDagContainer.innerHTML = "<p style='text-align:center; color: var(--text-muted-color); padding: 20px;'>Rendering graph...</p>"; // Loading message
        try {
            const definition = generateMermaidDefinition(graphData);
            // Ensure the container is visible and has dimensions before rendering for complex graphs
            if (graphDagViewContent.offsetParent === null) { // Check if hidden
                console.warn("Mermaid container is hidden, rendering might be suboptimal. Ensure tab is active.");
            }
            const { svg } = await mermaid.render('mermaidGeneratedSvg', definition);
            mermaidDagContainer.innerHTML = svg;
             // Add pan and zoom capabilities if desired (example using a simple library or custom code)
            // This is a placeholder for more advanced interaction.
            const svgElement = mermaidDagContainer.querySelector('svg');
            if (svgElement) {
                svgElement.style.cursor = 'grab';
                // Basic pan/zoom can be complex to implement robustly without a library.
            }

        } catch (error) {
            console.error("Mermaid rendering error:", error, "\nDefinition:", generateMermaidDefinition(graphData));
            mermaidDagContainer.textContent = "Error rendering DAG. Check console for details.";
        }
    }
    
    window.hideConfirmationModal = () => { // Make it globally accessible for the close button in HTML
        confirmationModal.style.display = 'none';
        currentConfirmationContext.graph2ThreadId = null;
        currentConfirmationContext.confirmationKey = null;
        currentConfirmationContext.operationId = null;
        currentConfirmationContext.effectiveNodeId = null;
        modalPayload.value = ''; 
    };

    function showConfirmationModal(details, graph2ThreadId) { 
        if (!details) {
            console.error("No details provided for confirmation modal.");
            addMessageToChat('System', 'Error: Missing details for confirmation modal.', 'error', 'system');
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


    modalConfirmButton.onclick = () => {
        if (!currentConfirmationContext.graph2ThreadId || !currentConfirmationContext.confirmationKey) {
            addMessageToChat('System', 'Error: Missing context for confirmation (Graph2 Thread ID or Confirmation Key). Cannot send resume command.', 'error', 'system');
            hideConfirmationModal();
            return;
        }

        let parsedPayload;
        try {
            if (modalPayload.value.trim() === "") {
                parsedPayload = {}; 
            } else {
                parsedPayload = JSON.parse(modalPayload.value);
            }
        } catch (e) {
            alert('Invalid JSON in payload textarea: ' + e.message);
            addMessageToChat('System', 'Payload in modal is not valid JSON. Please correct it.', 'error', 'system');
            return;
        }

        const resumeData = {
            confirmation_key: currentConfirmationContext.confirmationKey,
            decision: true,
            modified_payload: parsedPayload,
            operationId: currentConfirmationContext.operationId, 
            effectiveNodeId: currentConfirmationContext.effectiveNodeId
        };
        
        const wsMessage = `resume_exec ${currentConfirmationContext.graph2ThreadId} ${JSON.stringify(resumeData)}`;
        
        addMessageToChat(
            `You (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, 
            `Confirming: ${currentConfirmationContext.operationId || 'action'} with payload.`, 
            'user',
            'user_input' 
        );
        ws.send(wsMessage);
        addMessageToChat(
            `System (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, 
            `Confirmation sent for ${currentConfirmationContext.effectiveNodeId}. Waiting for workflow to resume...`, 
            'status', 
            'system'
        );
        hideConfirmationModal();
        showThinking(true); 
    };

    modalCancelButton.onclick = () => {
        if (!currentConfirmationContext.graph2ThreadId || !currentConfirmationContext.confirmationKey) {
            addMessageToChat('System', 'Error: Missing context for cancellation. Cannot send resume command.', 'error', 'system');
            hideConfirmationModal();
            return;
        }
        const resumeData = {
            confirmation_key: currentConfirmationContext.confirmationKey,
            decision: false, 
            operationId: currentConfirmationContext.operationId,
            effectiveNodeId: currentConfirmationContext.effectiveNodeId
        };
        const wsMessage = `resume_exec ${currentConfirmationContext.graph2ThreadId} ${JSON.stringify(resumeData)}`;
        
        addMessageToChat(
            `You (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, 
            `Cancelling confirmation for: ${currentConfirmationContext.operationId || 'action'}.`, 
            'user',
            'user_input'
        );
        ws.send(wsMessage);
         addMessageToChat(
            `System (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, 
            `Cancellation sent for ${currentConfirmationContext.effectiveNodeId}.`, 
            'status', 
            'system'
        );
        hideConfirmationModal();
    };

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

    showGraphTab('dag'); // MODIFIED: Default to DAG view
    connectWebSocket(); 
});
