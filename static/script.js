// OpenAPI Agent script.js

document.addEventListener('DOMContentLoaded', () => {
    console.log('[DEBUG] DOMContentLoaded event fired.');

    // DOM Elements
    const chatMessagesDiv = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
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

    // State Variables
    let ws;
    let currentGraphData = null;
    const initialSendButtonHTML = sendButton.innerHTML;
    let currentConfirmationContext = {
        graph2ThreadId: null,
        confirmationKey: null,
        operationId: null,
        effectiveNodeId: null
    };

    // --- Function Declarations ---

    /**
     * Updates the state of the send button and related UI elements.
     * @param {boolean} isThinking - True if the system is processing, false otherwise.
     */
    function updateSendButtonState(isThinking) {
        if (isThinking) {
            sendButton.classList.add('thinking');
            sendButton.innerHTML = `<div class="spinner" style="display: inline-block;"></div> Thinking...`;
            sendButton.disabled = true;
            userInput.disabled = true;
            runWorkflowButton.disabled = true;
        } else {
            sendButton.classList.remove('thinking');
            sendButton.innerHTML = initialSendButtonHTML;
            sendButton.disabled = false;
            userInput.disabled = false;
            runWorkflowButton.disabled = !currentGraphData;
        }
    }

    /**
     * Escapes HTML special characters in a string.
     * @param {*} unsafe - The input to escape.
     * @returns {string} The escaped string.
     */
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

    /**
     * Adds a message to the chat interface.
     * Implements basic appending for sequential "Planner" messages of non-terminal types.
     * @param {string} sender - The sender of the message.
     * @param {*} content - The content of the message.
     * @param {string} type - The type of the message.
     * @param {string} [sourceGraph="graph1_planning"] - The source of the message.
     */
    function addMessageToChat(sender, content, type, sourceGraph = "graph1_planning") {
        if (typeof addMessageToChat !== 'function') {
            console.error('[FATAL] addMessageToChat is not a function within its own body! This should not happen.');
            return;
        }
        // console.log(`[DEBUG] addMessageToChat called by: ${sender}, type: ${type}, source: ${sourceGraph}`);

        const lastMessageElement = chatMessagesDiv.lastElementChild;
        let shouldAppend = false;
        const appendablePlannerTypes = ["status", "intermediate", "info"];

        if (lastMessageElement &&
            lastMessageElement.dataset.source === "graph1_planning" &&
            sourceGraph === "graph1_planning" &&
            appendablePlannerTypes.includes(lastMessageElement.dataset.type) &&
            appendablePlannerTypes.includes(type) &&
            type !== "final" && type !== "error") {
            shouldAppend = true;
        }

        let messageContentText = "";
        if (typeof content === 'string') {
            messageContentText = content;
        } else if (typeof content === 'object' && content !== null && content.message) {
            messageContentText = content.message;
            if (content.details) {
                shouldAppend = false;
            }
        } else if (typeof content === 'object' && content !== null) {
            shouldAppend = false;
        } else {
            messageContentText = String(content);
        }

        if (shouldAppend && lastMessageElement) {
            const lastContentElement = lastMessageElement.querySelector('.message-content');
            if (lastContentElement) {
                const newParagraph = document.createElement('p');
                let formattedAppendedText = messageContentText.replace(/```([\s\S]*?)```/g, (match, code) => {
                    return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
                });
                formattedAppendedText = formattedAppendedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                formattedAppendedText = formattedAppendedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
                newParagraph.innerHTML = formattedAppendedText;
                lastContentElement.appendChild(newParagraph);
                lastMessageElement.dataset.type = type;
            }
        } else {
            const messageElement = document.createElement('div');
            let senderClass = 'agent-message';
            let senderName = sender;

            if (sender === 'You') {
                senderClass = 'user-message';
            } else if (sourceGraph === "graph2_execution") {
                senderClass = 'workflow-message';
                senderName = `Workflow (${(content && content.node_name) ? content.node_name : type})`;
            } else if (sourceGraph === "system" || sender === "System" || sourceGraph === "system_error" || sourceGraph === "system_warning" || sourceGraph === "system_critical") {
                senderClass = 'system-message';
                senderName = 'System';
            } else if (sender === 'Planner' || sourceGraph === 'graph1_planning') {
                senderClass = 'planner-message';
                senderName = 'Planner';
            }

            messageElement.classList.add('message', senderClass);
            messageElement.dataset.source = sourceGraph;
            messageElement.dataset.type = type;

            if (chatMessagesDiv.children.length > 1) {
                setTimeout(() => messageElement.classList.add('new-message-animation'), 50);
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
                formattedContent = formattedContent.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
                contentElement.innerHTML = formattedContent.split('\n').map(line => `<p>${line}</p>`).join('');
            } else if (typeof content === 'object' && content !== null) {
                if (sourceGraph === "graph2_execution") {
                    let detailsHtml = "";
                    if (content.node_name) detailsHtml += `<strong>Node:</strong> ${escapeHtml(content.node_name)}<br>`;
                    if (content.input_preview) detailsHtml += `Input Preview: <pre>${escapeHtml(content.input_preview)}</pre>`;
                    if (content.status_code) detailsHtml += `Status: <span class="status-code status-${String(content.status_code).charAt(0)}xx">${escapeHtml(content.status_code)}</span> `;
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

            if (type === 'error' || (typeof content === 'object' && content && content.error)) {
                if (sourceGraph === 'graph1_planning') {
                    let errorPrefix = `<strong style="color: var(--error-color); font-weight: bold;">Planner Alert:</strong> `;
                    let currentContentHTML = contentElement.innerHTML;
                    if (typeof content === 'object' && content.error) {
                        contentElement.innerHTML = errorPrefix + `<pre>${escapeHtml(content.error)}</pre>`;
                        if (content.message && content.message !== content.error) {
                           contentElement.innerHTML += `<p><em>Original message: ${escapeHtml(content.message)}</em></p>`;
                        }
                    } else {
                        contentElement.innerHTML = errorPrefix + currentContentHTML;
                    }
                } else {
                    messageElement.classList.add('error-message');
                }
            }
            messageElement.appendChild(senderElement);
            messageElement.appendChild(contentElement);
            chatMessagesDiv.appendChild(messageElement);
        }
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
    }
    console.log('[DEBUG] addMessageToChat defined, type:', typeof addMessageToChat);


    /**
     * Establishes and manages the WebSocket connection.
     */
    function connectWebSocket() {
        console.log('[DEBUG] connectWebSocket called. typeof addMessageToChat:', typeof addMessageToChat);
        if (typeof addMessageToChat !== 'function') {
            console.error('[FATAL] addMessageToChat is not available in connectWebSocket!');
            alert('Critical error: Chat functionality unavailable. Please refresh.');
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/openapi_agent`;

        addMessageToChat('System', `Attempting to connect to ${wsUrl}...`, 'info', 'system');
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            if (typeof addMessageToChat !== 'function') {
                 console.error('[FATAL] addMessageToChat is not available in ws.onopen!'); return;
            }
            addMessageToChat('System', 'Successfully connected to the agent.', 'info', 'system');
            updateSendButtonState(false);
            runWorkflowButton.disabled = true;
            updateGraphViewEmptyState(true);
        };

        ws.onmessage = (event) => {
            if (typeof addMessageToChat !== 'function') {
                console.error('[FATAL] addMessageToChat is not available in ws.onmessage!'); return;
            }
            try {
                const data = JSON.parse(event.data);
                console.log("WS RX:", data);

                let sender = 'Agent';
                let messageContent = data.content;
                let messageType = data.type;
                let sourceGraph = data.source || "unknown_source";

                if (sourceGraph === 'graph1_planning' && data.type === "graph_update") {
                    currentGraphData = data.content;
                    graphJsonViewPre.textContent = JSON.stringify(currentGraphData, null, 2);
                    addMessageToChat('System', 'Execution graph has been updated.', 'info', 'system');
                    updateGraphViewEmptyState(false);
                    if (graphDagViewContent.classList.contains('active')) {
                        renderMermaidGraphUI(currentGraphData);
                    }
                } else {
                    addMessageToChat(sender, messageContent, messageType, sourceGraph);
                }

                let shouldStopThinking = false;
                if (data.source === "graph1_planning") {
                    if (data.type === "final" || data.type === "error") {
                        shouldStopThinking = true;
                    }
                } else if (data.source === "graph2_execution") {
                    if (data.type === "execution_completed" ||
                        data.type === "execution_failed" ||
                        data.type === "workflow_timeout" ||
                        data.type === "error") {
                        shouldStopThinking = true;
                    }
                    if (data.type === "human_intervention_required") {
                        showConfirmationModal(data.content.details_for_ui, data.graph2_thread_id);
                    }
                } else if (data.source === "system_error" || data.source === "system_critical") {
                    shouldStopThinking = true;
                } else if (data.source === "system" && data.type === "info" && data.content && data.content.message === "Successfully connected to the agent.") {
                    shouldStopThinking = true;
                }

                if (shouldStopThinking) {
                    updateSendButtonState(false);
                } else if (data.type === "graph_update" && sourceGraph === "graph1_planning"){
                    runWorkflowButton.disabled = !currentGraphData;
                }

            } catch (error) {
                console.error("Error processing WebSocket message:", error, "Raw data:", event.data);
                addMessageToChat('System', 'Error processing message from server. Check console.', 'error', 'system');
                updateSendButtonState(false);
            }
        };

        ws.onerror = (error) => {
            if (typeof addMessageToChat !== 'function') {
                console.error('[FATAL] addMessageToChat is not available in ws.onerror!'); return;
            }
            console.error('WebSocket Error:', error);
            addMessageToChat('System', 'WebSocket connection error. Check console.', 'error', 'system');
            updateSendButtonState(false);
            runWorkflowButton.disabled = true;
        };

        ws.onclose = (event) => {
            if (typeof addMessageToChat !== 'function') {
                console.error('[FATAL] addMessageToChat is not available in ws.onclose!'); return;
            }
            addMessageToChat('System', `WebSocket disconnected. Code: ${event.code}. Attempting to reconnect in 5 seconds...`, 'system', 'system');
            updateSendButtonState(false);
            runWorkflowButton.disabled = true;
            setTimeout(connectWebSocket, 5000);
        };
    }

    /**
     * Updates the empty state message for graph views.
     * @param {boolean} isEmpty - True if no graph data is available.
     */
    function updateGraphViewEmptyState(isEmpty) {
        const dagTabPane = graphDagViewContent;
        const jsonTabPane = graphJsonViewContent;

        if (isEmpty) {
            dagTabPane.classList.add('is-empty');
            jsonTabPane.classList.add('is-empty');
            graphJsonViewPre.textContent = "JSON representation of the graph will appear here once a plan is generated.";
            graphJsonViewPre.classList.add('is-empty');
            renderMermaidGraphUI(null);
        } else {
            dagTabPane.classList.remove('is-empty');
            jsonTabPane.classList.remove('is-empty');
            graphJsonViewPre.classList.remove('is-empty');
        }
    }

    /**
     * Generates the Mermaid syntax definition for the graph.
     * @param {object|null} graph - The graph data object, or null if no graph.
     * @returns {string} The Mermaid graph definition string.
     */
    function generateMermaidDefinition(graph) {
        const emptyFill = '#fcfcfc';
        const emptyStroke = '#e0e0e0';
        const emptyColor = '#7f8c8d';
        const defaultNodeFill = '#ECEFF1';
        const defaultNodeStroke = '#90A4AE';
        const defaultNodeTextColor = '#37474F';
        const startEndFill = '#546E7A';
        const startEndStroke = '#37474F';
        const apiNodeFill = 'var(--primary-color)';
        const apiNodeStroke = 'var(--primary-hover-color)';
        const confirmedNodeFill = 'var(--success-color)';
        const confirmedNodeStroke = '#1E8E3E';
        const skippedNodeFill = 'var(--warning-color)';
        const skippedNodeStroke = '#F57F17';
        const skippedNodeTextColor = '#212121';
        const errorNodeFill = 'var(--error-color)';
        const errorNodeStroke = '#B71C1C';
        const interruptNodeFill = 'var(--accent-color)';
        const interruptNodeStroke = '#0D8A9F';

        if (!graph || !graph.nodes || !graph.edges) {
            return `graph TD\n    empty[<i class='fas fa-drafting-compass icon-placeholder'></i><br/>No Workflow Graph Available<br/><small>Please provide an OpenAPI specification and define a goal to generate a visual workflow.</small>];\n    classDef empty fill:${emptyFill},stroke:${emptyStroke},color:${emptyColor},padding:30px,font-style:italic,text-align:center,rx:12px,ry:12px,font-size:14px,border-width:1px,border-style:dashed;`;
        }

        let def = "graph TD;\n";
        def += `    classDef default fill:${defaultNodeFill},stroke:${defaultNodeStroke},stroke-width:1.5px,color:${defaultNodeTextColor},rx:6px,ry:6px,padding:10px 15px,font-size:13px,font-family:'Inter';\n`;
        def += `    classDef startEnd fill:${startEndFill},stroke:${startEndStroke},color:white,font-weight:bold;\n`;
        def += `    classDef apiNode fill:${apiNodeFill},stroke:${apiNodeStroke},color:white;\n`;
        def += `    classDef confirmedNode fill:${confirmedNodeFill},stroke:${confirmedNodeStroke},color:white;\n`;
        def += `    classDef skippedNode fill:${skippedNodeFill},stroke:${skippedNodeStroke},color:${skippedNodeTextColor};\n`;
        def += `    classDef errorNode fill:${errorNodeFill},stroke:${errorNodeStroke},color:white;\n`;
        def += `    classDef interruptNode fill:${interruptNodeFill},stroke:${interruptNodeStroke},color:white;\n`;

        const sanitizeId = (id) => String(id).replace(/[^a-zA-Z0-9_]/g, '_');

        graph.nodes.forEach(node => {
            const id = sanitizeId(node.effective_id || node.operationId);
            let labelText = node.summary || node.operationId;
            if (node.summary && node.summary !== node.operationId && node.operationId !== "START_NODE" && node.operationId !== "END_NODE") {
                labelText = `${escapeHtml(node.summary)}<br/><small>(${escapeHtml(node.operationId)} / ${escapeHtml(node.effective_id)})</small>`;
            } else {
                labelText = `<b>${escapeHtml(node.operationId)}</b><br/><small>(${escapeHtml(node.effective_id || node.operationId)})</small>`;
            }
            def += `    ${id}("${labelText.replace(/"/g, '#quot;')}");\n`;
            let nodeClass = 'apiNode';
            if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                nodeClass = 'startEnd';
            } else if (node.requires_confirmation) {
                 nodeClass = 'interruptNode';
            }
            def += `    class ${id} ${nodeClass};\n`;
        });

        graph.edges.forEach(edge => {
            const from = sanitizeId(edge.from_node);
            const to = sanitizeId(edge.to_node);
            const label = edge.description ? `|"${escapeHtml(edge.description.substring(0, 50)).replace(/"/g, '#quot;')}"|` : "";
            def += `    ${from} -->${label} ${to};\n`;
        });
        // console.log("Generated Mermaid Definition:\n", def); // Keep for debugging if needed
        return def;
    }

    /**
     * Renders the Mermaid graph in the UI.
     * @param {object|null} graphData - The graph data object, or null.
     */
    async function renderMermaidGraphUI(graphData) {
        if (typeof mermaid === 'undefined') {
            mermaidDagContainer.innerHTML = "<div class='mermaid-placeholder error'><i class='fas fa-exclamation-triangle'></i><p>Mermaid.js library not loaded.</p><p>Cannot render graph visualization.</p></div>";
            return;
        }
        const definition = generateMermaidDefinition(graphData);
        if (!graphData) {
            try {
                mermaidDagContainer.classList.add('rendering-placeholder');
                const { svg } = await mermaid.render('mermaidGeneratedSvgPlaceholder', definition);
                mermaidDagContainer.innerHTML = svg;
                mermaidDagContainer.classList.remove('rendering-placeholder');
                const placeholderSvg = mermaidDagContainer.querySelector('svg');
                if(placeholderSvg) {
                    placeholderSvg.style.maxWidth = '350px';
                    placeholderSvg.style.maxHeight = '250px';
                    placeholderSvg.style.margin = 'auto';
                }
            } catch (error) {
                 console.error("Mermaid rendering error for placeholder:", error, "\nDefinition:", definition);
                 mermaidDagContainer.innerHTML = "<div class='mermaid-placeholder error'><i class='fas fa-exclamation-triangle'></i><p>Error rendering placeholder.</p><p>Check console for details.</p></div>";
            }
            return;
        }
        mermaidDagContainer.innerHTML = "<div class='mermaid-placeholder loading'><i class='fas fa-spinner fa-spin'></i><p>Rendering graph...</p></div>";
        try {
            if (graphDagViewContent.offsetParent === null) {
                console.warn("Mermaid container is hidden, rendering might be suboptimal. Ensure tab is active.");
            }
            const { svg } = await mermaid.render('mermaidGeneratedSvg', definition);
            mermaidDagContainer.innerHTML = svg;
            const svgElement = mermaidDagContainer.querySelector('svg');
            if (svgElement) {
                svgElement.style.cursor = 'grab';
            }
        } catch (error) {
            console.error("Mermaid rendering error:", error, "\nProblematic Definition:", definition);
            mermaidDagContainer.innerHTML = "<div class='mermaid-placeholder error'><i class='fas fa-exclamation-triangle'></i><p>Error rendering DAG.</p><p>Check console for details and the problematic definition.</p></div>";
        }
    }

    /**
     * Shows the confirmation modal with details of the action.
     * @param {object} details - Details for the confirmation (operationId, path, payload, etc.).
     * @param {string} graph2ThreadId - The thread ID of the Graph 2 execution.
     */
    function showConfirmationModal(details, graph2ThreadId) {
        if (typeof addMessageToChat !== 'function') {
            console.error('[FATAL] addMessageToChat is not available in showConfirmationModal!'); return;
        }
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
                payloadToDisplay = "Error: Could not format payload for display.";
                console.error("Error stringifying payload_to_confirm:", details.payload_to_confirm, e);
            }
        }
        modalPayload.value = payloadToDisplay;
        confirmationModal.style.display = 'flex';
    }


    // --- Global Assignments & Event Listeners ---

    window.sendMessage = () => {
        if (typeof addMessageToChat !== 'function') {
            console.error('[FATAL] addMessageToChat is not available in window.sendMessage!'); return;
        }
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
        updateSendButtonState(true);
    };

    window.runCurrentWorkflow = () => {
        if (typeof addMessageToChat !== 'function') {
            console.error('[FATAL] addMessageToChat is not available in window.runCurrentWorkflow!'); return;
        }
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
        updateSendButtonState(true);
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
        updateGraphViewEmptyState(!currentGraphData);
    };

    window.hideConfirmationModal = () => {
        confirmationModal.style.display = 'none';
        currentConfirmationContext.graph2ThreadId = null;
        currentConfirmationContext.confirmationKey = null;
        currentConfirmationContext.operationId = null;
        currentConfirmationContext.effectiveNodeId = null;
        modalPayload.value = '';
    };

    modalConfirmButton.onclick = () => {
        if (typeof addMessageToChat !== 'function') {
            console.error('[FATAL] addMessageToChat is not available in modalConfirmButton.onclick!'); return;
        }
        if (!currentConfirmationContext.graph2ThreadId || !currentConfirmationContext.confirmationKey) {
            addMessageToChat('System', 'Error: Missing context for confirmation. Cannot send resume command.', 'error', 'system');
            hideConfirmationModal();
            return;
        }
        let parsedPayload;
        try {
            parsedPayload = modalPayload.value.trim() === "" ? {} : JSON.parse(modalPayload.value);
        } catch (e) {
            alert('Invalid JSON in payload textarea: ' + e.message);
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
        addMessageToChat(`You (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, `Confirming: ${currentConfirmationContext.operationId || 'action'}`, 'user', 'user_input');
        ws.send(wsMessage);
        hideConfirmationModal();
    };

    modalCancelButton.onclick = () => {
        if (typeof addMessageToChat !== 'function') {
            console.error('[FATAL] addMessageToChat is not available in modalCancelButton.onclick!'); return;
        }
        if (!currentConfirmationContext.graph2ThreadId || !currentConfirmationContext.confirmationKey) {
            addMessageToChat('System', 'Error: Missing context for cancellation.', 'error', 'system');
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
        addMessageToChat(`You (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, `Cancelling confirmation for: ${currentConfirmationContext.operationId || 'action'}.`, 'user', 'user_input');
        ws.send(wsMessage);
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

    // Initial Setup Calls
    showGraphTab('dag');
    connectWebSocket(); // This will call addMessageToChat
}); 
