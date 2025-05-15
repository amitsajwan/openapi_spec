// OpenAPI Agent script.js

document.addEventListener('DOMContentLoaded', () => {
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

    let ws;
    let currentGraphData = null;
    const initialSendButtonHTML = sendButton.innerHTML;

    let currentConfirmationContext = {
        graph2ThreadId: null,
        confirmationKey: null,
        operationId: null,
        effectiveNodeId: null
    };

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
            runWorkflowButton.disabled = true; // Also disable run workflow button when thinking
        } else {
            sendButton.classList.remove('thinking');
            sendButton.innerHTML = initialSendButtonHTML;
            sendButton.disabled = false;
            userInput.disabled = false;
            // Only enable runWorkflowButton if a graph is loaded
            runWorkflowButton.disabled = !currentGraphData;
        }
    }

    /**
     * Escapes HTML special characters in a string.
     * If input is not a string, attempts to stringify it (e.g., for objects).
     * @param {*} unsafe - The input to escape.
     * @returns {string} The escaped string.
     */
    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            if (unsafe === null || unsafe === undefined) return '';
            try {
                return JSON.stringify(unsafe, null, 2); // Pretty print objects/arrays
            } catch (e) {
                return String(unsafe); // Fallback for non-serializable objects
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
     * @param {string} sender - The sender of the message (e.g., "You", "Agent", "Planner").
     * @param {*} content - The content of the message (string or object).
     * @param {string} type - The type of the message (e.g., "user", "final", "error").
     * @param {string} [sourceGraph="graph1_planning"] - The source of the message.
     */
    function addMessageToChat(sender, content, type, sourceGraph = "graph1_planning") {
        const messageElement = document.createElement('div');
        let senderClass = 'agent-message'; // Default
        let senderName = sender;

        // Determine sender class and name based on source and sender
        if (sender === 'You') {
            senderClass = 'user-message';
        } else if (sourceGraph === "graph2_execution") {
            senderClass = 'workflow-message';
            // Make sender name more specific for workflow messages based on type
            senderName = `Workflow (${content.node_name || type})`;
        } else if (sourceGraph === "system" || sender === "System" || sourceGraph === "system_error" || sourceGraph === "system_warning" || sourceGraph === "system_critical") {
            senderClass = 'system-message';
            senderName = 'System';
        } else if (sender === 'Planner' || sourceGraph === 'graph1_planning') {
            senderClass = 'planner-message';
            senderName = 'Planner';
        }

        messageElement.classList.add('message', senderClass);
        // Add animation for new messages
        if (chatMessagesDiv.children.length > 1) {
             setTimeout(() => messageElement.classList.add('new-message-animation'), 50);
        }

        const senderElement = document.createElement('div');
        senderElement.classList.add('message-sender');
        senderElement.textContent = senderName;

        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');

        // Format message content
        if (typeof content === 'string') {
            // Basic Markdown-like formatting for strings
            let formattedContent = content.replace(/```([\s\S]*?)```/g, (match, code) => {
                return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
            });
            formattedContent = formattedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedContent = formattedContent.replace(/\*(.*?)\*/g, '<em>$1</em>');
            formattedContent = formattedContent.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
            contentElement.innerHTML = formattedContent.split('\n').map(line => `<p>${line}</p>`).join('');
        } else if (typeof content === 'object' && content !== null) {
            // Handle object content, especially for graph2_execution messages
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
            } else if (content.message) { // For planner or other object messages with a 'message' field
                contentElement.innerHTML = `<p>${escapeHtml(content.message)}</p>`;
                 if (content.details) {
                    contentElement.innerHTML += `<pre>${escapeHtml(content.details)}</pre>`;
                 }
            } else { // Fallback for other objects
                contentElement.innerHTML = `<pre>${escapeHtml(content)}</pre>`;
            }
        } else {
             contentElement.innerHTML = `<p>${escapeHtml(String(content))}</p>`; // Fallback for other types
        }

        // Apply error styling based on type and content
        if (type === 'error' || (typeof content === 'object' && content && content.error)) {
            if (sourceGraph === 'graph1_planning') { // Planner-specific error display
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
                // Note: Planner errors use planner-message background, not full error-message red.
            } else {
                // For system, workflow errors, apply the full error-message class
                messageElement.classList.add('error-message');
            }
        }

        messageElement.appendChild(senderElement);
        messageElement.appendChild(contentElement);
        chatMessagesDiv.appendChild(messageElement);
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight; // Auto-scroll to the latest message
    }

    /**
     * Establishes and manages the WebSocket connection.
     */
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/openapi_agent`;

        addMessageToChat('System', `Attempting to connect to ${wsUrl}...`, 'info', 'system');
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            addMessageToChat('System', 'Successfully connected to the agent.', 'info', 'system');
            updateSendButtonState(false); // Stop thinking on successful connection
            runWorkflowButton.disabled = true; // Workflow button disabled until a graph is loaded
            updateGraphViewEmptyState(true); // Initialize graph view as empty
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("WS RX:", data); // Log received data for debugging

                let sender = 'Agent'; // Default sender
                let messageContent = data.content;
                let messageType = data.type;
                let sourceGraph = data.source || "unknown_source";

                // Determine sender and content based on message source and type
                if (sourceGraph === 'graph1_planning') {
                    sender = 'Planner';
                    if (data.type === "graph_update") {
                        currentGraphData = data.content;
                        graphJsonViewPre.textContent = JSON.stringify(currentGraphData, null, 2);
                        addMessageToChat('System', 'Execution graph has been updated.', 'info', 'system');
                        updateGraphViewEmptyState(false);
                        if (graphDagViewContent.classList.contains('active')) {
                            renderMermaidGraphUI(currentGraphData);
                        }
                        runWorkflowButton.disabled = !currentGraphData; // Enable/disable based on graph presence
                        // Do not stop thinking for graph_update, as more planning might follow
                    } else {
                        // For other planner messages, content might be an object with a 'message' field
                        messageContent = (typeof data.content === 'object' && data.content !== null && data.content.message) ? data.content.message : data.content;
                        if (typeof data.content === 'object' && data.content !== null && data.content.error && !messageContent){
                            messageContent = data.content.error; // If error is the primary content
                        }
                        addMessageToChat(sender, messageContent, messageType, sourceGraph);
                    }
                } else if (sourceGraph === 'graph2_execution') {
                    // For workflow messages, sender name is derived in addMessageToChat
                    addMessageToChat(sender, messageContent, messageType, sourceGraph); // Pass full content object
                    if (data.type === "human_intervention_required") {
                        showConfirmationModal(data.content.details_for_ui, data.graph2_thread_id);
                        // Do not stop thinking here as the main planning might still be active, or user might interact further.
                    }
                } else if (sourceGraph === 'system' || sourceGraph === 'system_error' || sourceGraph === 'system_warning' || sourceGraph === 'system_critical') {
                    sender = 'System';
                    messageContent = data.content.error || data.content.message || data.content;
                    addMessageToChat(sender, messageContent, messageType, sourceGraph);
                } else {
                    // Fallback for unknown sources
                     addMessageToChat(sender, messageContent, messageType, sourceGraph);
                }


                // --- MODIFIED LOGIC FOR STOPPING THE "THINKING" INDICATOR ---
                let shouldStopThinking = false;

                if (data.source === "graph1_planning") {
                    // Stop thinking if Graph 1 sends a "final" message or an "error"
                    if (data.type === "final" || data.type === "error") {
                        shouldStopThinking = true;
                    }
                } else if (data.source === "graph2_execution") {
                    // Stop thinking if Graph 2 completes, fails, or times out
                    if (data.type === "execution_completed" ||
                        data.type === "execution_failed" ||
                        data.type === "workflow_timeout" ||
                        data.type === "error") { // Assuming 'error' from G2 is also terminal for that G2 attempt
                        shouldStopThinking = true;
                    }
                } else if (data.source === "system_error" || data.source === "system_critical") {
                    // Stop thinking for critical system errors
                    shouldStopThinking = true;
                } else if (data.source === "system" && data.type === "info" && data.content && data.content.message === "Successfully connected to the agent.") {
                    // Initial connection success should also stop any lingering "thinking" from page load.
                    shouldStopThinking = true;
                }


                if (shouldStopThinking) {
                    updateSendButtonState(false);
                }

            } catch (error) {
                console.error("Error processing WebSocket message:", error, "Raw data:", event.data);
                addMessageToChat('System', 'Error processing message from server. Check console.', 'error', 'system');
                updateSendButtonState(false); // Fallback: stop thinking on client-side error
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            addMessageToChat('System', 'WebSocket connection error. Check console.', 'error', 'system');
            updateSendButtonState(false); // Stop thinking on WebSocket error
            runWorkflowButton.disabled = true;
        };

        ws.onclose = (event) => {
            addMessageToChat('System', `WebSocket disconnected. Code: ${event.code}. Attempting to reconnect in 5 seconds...`, 'system', 'system');
            updateSendButtonState(false); // Stop thinking on WebSocket close
            runWorkflowButton.disabled = true;
            setTimeout(connectWebSocket, 5000); // Attempt to reconnect
        };
    }

    /**
     * Sends the user's message via WebSocket.
     */
    window.sendMessage = () => {
        const messageText = userInput.value.trim();
        if (!messageText) return;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessageToChat('System', 'Not connected. Please wait or try refreshing.', 'error', 'system');
            return;
        }
        addMessageToChat('You', messageText, 'user', 'user_input');
        ws.send(messageText);
        userInput.value = ''; // Clear input field
        userInput.style.height = 'auto'; // Reset textarea height
        updateSendButtonState(true); // Start thinking
    };

    /**
     * Sends a command to run the current workflow.
     */
    window.runCurrentWorkflow = () => {
        if (!currentGraphData) {
            addMessageToChat('System', 'No workflow graph loaded to run.', 'error', 'system');
            return;
        }
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessageToChat('System', 'Not connected. Cannot run workflow.', 'error', 'system');
            return;
        }
        const command = "run workflow"; // This command is interpreted by the backend router
        addMessageToChat('You', command, 'user', 'user_input');
        ws.send(command);
        // Don't add a system message here immediately, let the backend confirm.
        updateSendButtonState(true); // Start thinking
        // runWorkflowButton will be disabled by updateSendButtonState(true)
    };

    /**
     * Updates the empty state message for graph views.
     * @param {boolean} isEmpty - True if no graph data is available.
     */
    function updateGraphViewEmptyState(isEmpty) {
        const dagTabPane = graphDagViewContent;
        const jsonTabPane = graphJsonViewContent;

        if (isEmpty) {
            dagTabPane.classList.add('is-empty');
            jsonTabPane.classList.add('is-empty'); // Ensure pre tag is also styled as empty
            graphJsonViewPre.textContent = "JSON representation of the graph will appear here once a plan is generated.";
            graphJsonViewPre.classList.add('is-empty'); // Add class for specific styling if needed
            renderMermaidGraphUI(null); // Render placeholder for Mermaid
        } else {
            dagTabPane.classList.remove('is-empty');
            jsonTabPane.classList.remove('is-empty');
            graphJsonViewPre.classList.remove('is-empty');
        }
    }

    /**
     * Switches between graph view tabs (JSON or DAG).
     * @param {string} tabName - The name of the tab to show ('json' or 'dag').
     */
    window.showGraphTab = (tabName) => {
        graphJsonViewContent.style.display = 'none';
        graphDagViewContent.style.display = 'none';
        graphJsonViewContent.classList.remove('active');
        graphDagViewContent.classList.remove('active');
        jsonTabButton.classList.remove('active');
        dagTabButton.classList.remove('active');

        if (tabName === 'json') {
            graphJsonViewContent.style.display = 'flex'; // Use flex for pre tag to grow
            graphJsonViewContent.classList.add('active');
            jsonTabButton.classList.add('active');
        } else if (tabName === 'dag') {
            graphDagViewContent.style.display = 'flex'; // Use flex for mermaid container
            graphDagViewContent.classList.add('active');
            dagTabButton.classList.add('active');
            renderMermaidGraphUI(currentGraphData); // Re-render if switching to DAG
        }
        updateGraphViewEmptyState(!currentGraphData); // Update empty state based on current data
    };

    /**
     * Generates the Mermaid syntax definition for the graph.
     * @param {object|null} graph - The graph data object, or null if no graph.
     * @returns {string} The Mermaid graph definition string.
     */
    function generateMermaidDefinition(graph) {
        // Theme variables for Mermaid nodes
        const emptyFill = '#fcfcfc';
        const emptyStroke = '#e0e0e0';
        const emptyColor = '#7f8c8d';

        const defaultNodeFill = '#ECEFF1'; // Light grey
        const defaultNodeStroke = '#90A4AE'; // Medium grey
        const defaultNodeTextColor = '#37474F'; // Dark grey text

        const startEndFill = '#546E7A'; // Dark blue-grey for start/end
        const startEndStroke = '#37474F';
        const apiNodeFill = 'var(--primary-color)'; // Use CSS variable for primary color
        const apiNodeStroke = 'var(--primary-hover-color)';
        const confirmedNodeFill = 'var(--success-color)'; // Green for confirmed
        const confirmedNodeStroke = '#1E8E3E';
        const skippedNodeFill = 'var(--warning-color)'; // Yellow for skipped
        const skippedNodeStroke = '#F57F17';
        const skippedNodeTextColor = '#212121'; // Dark text for yellow background
        const errorNodeFill = 'var(--error-color)'; // Red for error
        const errorNodeStroke = '#B71C1C';
        const interruptNodeFill = 'var(--accent-color)'; // Teal for interrupt/confirmation needed
        const interruptNodeStroke = '#0D8A9F';

        // Base definition for an empty graph or placeholder
        if (!graph || !graph.nodes || !graph.edges) {
            return `graph TD\n    empty[<i class='fas fa-drafting-compass icon-placeholder'></i><br/>No Workflow Graph Available<br/><small>Please provide an OpenAPI specification and define a goal to generate a visual workflow.</small>];\n    classDef empty fill:${emptyFill},stroke:${emptyStroke},color:${emptyColor},padding:30px,font-style:italic,text-align:center,rx:12px,ry:12px,font-size:14px,border-width:1px,border-style:dashed;`;
        }

        let def = "graph TD;\n"; // Top-Down direction for the graph
        // Define CSS classes for different node types within Mermaid
        def += `    classDef default fill:${defaultNodeFill},stroke:${defaultNodeStroke},stroke-width:1.5px,color:${defaultNodeTextColor},rx:6px,ry:6px,padding:10px 15px,font-size:13px,font-family:'Inter';\n`;
        def += `    classDef startEnd fill:${startEndFill},stroke:${startEndStroke},color:white,font-weight:bold;\n`;
        def += `    classDef apiNode fill:${apiNodeFill},stroke:${apiNodeStroke},color:white;\n`;
        def += `    classDef confirmedNode fill:${confirmedNodeFill},stroke:${confirmedNodeStroke},color:white;\n`;
        def += `    classDef skippedNode fill:${skippedNodeFill},stroke:${skippedNodeStroke},color:${skippedNodeTextColor};\n`;
        def += `    classDef errorNode fill:${errorNodeFill},stroke:${errorNodeStroke},color:white;\n`;
        def += `    classDef interruptNode fill:${interruptNodeFill},stroke:${interruptNodeStroke},color:white;\n`;

        const sanitizeId = (id) => String(id).replace(/[^a-zA-Z0-9_]/g, '_'); // Sanitize IDs for Mermaid

        // Add nodes to the definition
        graph.nodes.forEach(node => {
            const id = sanitizeId(node.effective_id || node.operationId);
            let labelText = node.summary || node.operationId;
            // Create more detailed labels for API nodes
            if (node.summary && node.summary !== node.operationId && node.operationId !== "START_NODE" && node.operationId !== "END_NODE") {
                labelText = `${escapeHtml(node.summary)}<br/><small>(${escapeHtml(node.operationId)} / ${escapeHtml(node.effective_id)})</small>`;
            } else {
                labelText = `<b>${escapeHtml(node.operationId)}</b><br/><small>(${escapeHtml(node.effective_id)})</small>`;
            }

            def += `    ${id}("${labelText.replace(/"/g, '#quot;')}");\n`; // Add node with label (escape quotes)

            // Assign class based on node type/status
            let nodeClass = 'apiNode'; // Default for API nodes
            if (node.operationId === "START_NODE" || node.operationId === "END_NODE") {
                nodeClass = 'startEnd';
            } else if (node.requires_confirmation) { // Mark nodes that need confirmation
                 nodeClass = 'interruptNode';
            }
            // Future: Add classes for 'confirmedNode', 'skippedNode', 'errorNode' based on execution status if available
            def += `    class ${id} ${nodeClass};\n`;
        });

        // Add edges to the definition
        graph.edges.forEach(edge => {
            const from = sanitizeId(edge.from_node);
            const to = sanitizeId(edge.to_node);
            // Add edge description if available
            const label = edge.description ? `|"${escapeHtml(edge.description.substring(0, 50)).replace(/"/g, '#quot;')}"|` : "";
            def += `    ${from} -->${label} ${to};\n`;
        });
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

        // Handle empty/placeholder graph rendering
        if (!graphData) {
            try {
                mermaidDagContainer.classList.add('rendering-placeholder');
                const { svg } = await mermaid.render('mermaidGeneratedSvgPlaceholder', definition);
                mermaidDagContainer.innerHTML = svg;
                mermaidDagContainer.classList.remove('rendering-placeholder');
                const placeholderSvg = mermaidDagContainer.querySelector('svg');
                if(placeholderSvg) { // Style the placeholder SVG
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

        // Render actual graph
        mermaidDagContainer.innerHTML = "<div class='mermaid-placeholder loading'><i class='fas fa-spinner fa-spin'></i><p>Rendering graph...</p></div>";
        try {
            // Ensure container is visible for correct rendering, though Mermaid 10 is better at this
            if (graphDagViewContent.offsetParent === null) {
                console.warn("Mermaid container is hidden, rendering might be suboptimal. Ensure tab is active.");
            }
            const { svg } = await mermaid.render('mermaidGeneratedSvg', definition);
            mermaidDagContainer.innerHTML = svg;
            const svgElement = mermaidDagContainer.querySelector('svg');
            if (svgElement) { // Make graph pannable/zoomable (basic cursor hint)
                svgElement.style.cursor = 'grab';
            }

        } catch (error) {
            console.error("Mermaid rendering error:", error, "\nDefinition:", definition);
            mermaidDagContainer.innerHTML = "<div class='mermaid-placeholder error'><i class='fas fa-exclamation-triangle'></i><p>Error rendering DAG.</p><p>Check console for details.</p></div>";
        }
    }

    /**
     * Hides the confirmation modal and resets its context.
     */
    window.hideConfirmationModal = () => {
        confirmationModal.style.display = 'none';
        // Reset context to avoid unintended reuse
        currentConfirmationContext.graph2ThreadId = null;
        currentConfirmationContext.confirmationKey = null;
        currentConfirmationContext.operationId = null;
        currentConfirmationContext.effectiveNodeId = null;
        modalPayload.value = ''; // Clear payload textarea
    };

    /**
     * Shows the confirmation modal with details of the action.
     * @param {object} details - Details for the confirmation (operationId, path, payload, etc.).
     * @param {string} graph2ThreadId - The thread ID of the Graph 2 execution.
     */
    function showConfirmationModal(details, graph2ThreadId) {
        if (!details) {
            console.error("No details provided for confirmation modal.");
            addMessageToChat('System', 'Error: Missing details for confirmation modal.', 'error', 'system');
            return;
        }

        // Store context for when the user confirms/cancels
        currentConfirmationContext.graph2ThreadId = graph2ThreadId;
        currentConfirmationContext.confirmationKey = details.confirmation_key; // Key to send back
        currentConfirmationContext.operationId = details.operationId;
        currentConfirmationContext.effectiveNodeId = details.effective_node_id;

        // Populate modal fields
        modalTitle.textContent = details.prompt || `Confirm API Call: ${details.operationId}`;
        modalOperationId.textContent = details.operationId || 'N/A';
        modalEffectiveNodeId.textContent = details.effective_node_id || 'N/A';
        modalMethod.textContent = details.method || 'N/A';
        modalPath.textContent = details.path || 'N/A';
        modalGraph2ThreadId.textContent = graph2ThreadId || 'N/A';

        let payloadToDisplay = "";
        if (details.payload_to_confirm !== undefined && details.payload_to_confirm !== null) {
            try {
                payloadToDisplay = JSON.stringify(details.payload_to_confirm, null, 2); // Pretty print JSON
            } catch (e) {
                payloadToDisplay = "Error: Could not format payload for display.";
                console.error("Error stringifying payload_to_confirm:", details.payload_to_confirm, e);
            }
        }
        modalPayload.value = payloadToDisplay;
        confirmationModal.style.display = 'flex'; // Show the modal
    }

    // Event listener for the modal's confirm button
    modalConfirmButton.onclick = () => {
        if (!currentConfirmationContext.graph2ThreadId || !currentConfirmationContext.confirmationKey) {
            addMessageToChat('System', 'Error: Missing context for confirmation. Cannot send resume command.', 'error', 'system');
            hideConfirmationModal();
            return;
        }
        let parsedPayload;
        try {
            // Parse the payload from the textarea (user might have edited it)
            parsedPayload = modalPayload.value.trim() === "" ? {} : JSON.parse(modalPayload.value);
        } catch (e) {
            alert('Invalid JSON in payload textarea: ' + e.message); // Simple alert for invalid JSON
            return;
        }
        // Prepare data to send back to resume the workflow
        const resumeData = {
            confirmation_key: currentConfirmationContext.confirmationKey,
            decision: true, // User confirmed
            modified_payload: parsedPayload,
            operationId: currentConfirmationContext.operationId,
            effectiveNodeId: currentConfirmationContext.effectiveNodeId
        };
        // Construct WebSocket message
        const wsMessage = `resume_exec ${currentConfirmationContext.graph2ThreadId} ${JSON.stringify(resumeData)}`;

        addMessageToChat(`You (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, `Confirming: ${currentConfirmationContext.operationId || 'action'}`, 'user', 'user_input');
        ws.send(wsMessage);
        // Don't add a system message "Confirmation sent" here; let the backend confirm receipt and processing.
        hideConfirmationModal();
        // Do not call updateSendButtonState(true) here, as this is a G2 interaction.
        // The main G1 send button's state should not be affected by G2 modal confirmation.
    };

    // Event listener for the modal's cancel button
    modalCancelButton.onclick = () => {
        if (!currentConfirmationContext.graph2ThreadId || !currentConfirmationContext.confirmationKey) {
            addMessageToChat('System', 'Error: Missing context for cancellation.', 'error', 'system');
            hideConfirmationModal();
            return;
        }
        // Prepare data indicating cancellation
        const resumeData = {
            confirmation_key: currentConfirmationContext.confirmationKey,
            decision: false, // User cancelled
            operationId: currentConfirmationContext.operationId,
            effectiveNodeId: currentConfirmationContext.effectiveNodeId
        };
        const wsMessage = `resume_exec ${currentConfirmationContext.graph2ThreadId} ${JSON.stringify(resumeData)}`;
        addMessageToChat(`You (to Workflow ${currentConfirmationContext.graph2ThreadId.slice(-4)})`, `Cancelling confirmation for: ${currentConfirmationContext.operationId || 'action'}.`, 'user', 'user_input');
        ws.send(wsMessage);
        hideConfirmationModal();
    };

    // Auto-resize textarea for user input
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });

    // Send message on Enter key (but not Shift+Enter for new lines)
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default Enter behavior (new line)
            sendMessage();
        }
    });

    // Initial setup
    showGraphTab('dag'); // Show DAG view by default
    connectWebSocket(); // Establish WebSocket connection on load
});
