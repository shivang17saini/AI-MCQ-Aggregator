document.addEventListener('DOMContentLoaded', () => {
    const analyzeButton = document.getElementById('analyze-button');
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    const answersGrid = document.getElementById('answers-grid');
    const consensusResults = document.getElementById('consensus-results');
    const errorBox = document.getElementById('error-box');
    const loadingIndicator = document.getElementById('loading-indicator');

    analyzeButton.addEventListener('click', async () => {
        const text = textInput.value;
        const image = imageInput.files[0];

        if (!text.trim() && !image) {
            showError("Please provide text or an image.");
            return;
        }

        const formData = new FormData();
        if (text.trim()) {
            formData.append('text', text);
        }
        if (image) {
            formData.append('image', image);
        }

        // Reset UI
        hideError();
        answersGrid.innerHTML = '';
        consensusResults.innerHTML = '';
        loadingIndicator.classList.remove('hidden');
        analyzeButton.disabled = true;

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                showError(errorData.error || 'An unknown error occurred.');
                stopLoading();
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            function processStream() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        stopLoading();
                        return;
                    }
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n\n');

                    lines.forEach(line => {
                        if (line.startsWith('data:')) {
                            const jsonData = line.substring(5);
                            try {
                                const event = JSON.parse(jsonData);
                                handleServerEvent(event);
                            } catch (e) {
                                console.error('Error parsing JSON:', jsonData, e);
                            }
                        }
                    });

                    processStream();
                }).catch(error => {
                    console.error('Stream reading error:', error);
                    showError('Failed to read analysis results.');
                    stopLoading();
                });
            }
            processStream();

        } catch (error) {
            console.error('Fetch error:', error);
            showError('Failed to connect to the server.');
            stopLoading();
        }
    });

    function handleServerEvent(event) {
        if (event.type === 'model_result') {
            addAnswerCard(event.data);
        } else if (event.type === 'consensus_result') {
            displayConsensus(event.data);
        } else if (event.type === 'done') {
            stopLoading();
        } else if (event.type === 'error') {
            showError(event.data.message);
        }
    }

    function stopLoading() {
        loadingIndicator.classList.add('hidden');
        analyzeButton.disabled = false;
    }
    
    function addAnswerCard(data) {
        const card = document.createElement('div');
        card.className = 'answer-card';

        const header = document.createElement('div');
        header.className = 'card-header';
        
        const modelName = document.createElement('h3');
        modelName.textContent = data.model;
        
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-btn';
        copyButton.textContent = 'Copy';
        copyButton.addEventListener('click', () => {
            const textarea = document.createElement('textarea');
            textarea.value = data.answer;
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                copyButton.textContent = 'Copied!';
            } catch (err) {
                console.error('Fallback: Oops, unable to copy', err);
                copyButton.textContent = 'Error';
            }
            document.body.removeChild(textarea);

            setTimeout(() => { copyButton.textContent = 'Copy'; }, 2000);
        });

        header.appendChild(modelName);
        header.appendChild(copyButton);

        const answerContent = document.createElement('div');
        answerContent.className = 'card-content';
        answerContent.innerHTML = formatAnswer(data.answer);

        card.appendChild(header);
        card.appendChild(answerContent);
        answersGrid.appendChild(card);
    }

    function displayConsensus(data) {
        consensusResults.innerHTML = '';
        if (data.length === 0) {
            const noConsensus = document.createElement('p');
            noConsensus.textContent = 'No consensus could be reached.';
            consensusResults.appendChild(noConsensus);
            return;
        }

        data.forEach(questionConsensus => {
            const questionContainer = document.createElement('div');
            questionContainer.className = 'consensus-question';

            const questionTitle = document.createElement('h4');
            questionTitle.textContent = `Question ${questionConsensus.question}`;
            questionContainer.appendChild(questionTitle);
            
            const votes = questionConsensus.votes;
            const sortedVotes = Object.entries(votes).sort(([, a], [, b]) => parseInt(b) - parseInt(a));

            if (sortedVotes.length > 0) {
                sortedVotes.forEach(([option, percentage], index) => {
                    const barContainer = document.createElement('div');
                    barContainer.className = 'consensus-bar-container';
                    
                    const label = document.createElement('span');
                    label.textContent = `Option ${option}: ${percentage}`;
                    
                    const bar = document.createElement('div');
                    bar.className = 'consensus-bar';
                    if (index === 0) {
                        bar.classList.add('winner');
                    }
                    bar.style.width = percentage;

                    barContainer.appendChild(label);
                    barContainer.appendChild(bar);
                    questionContainer.appendChild(barContainer);
                });
            } else {
                 const noVotes = document.createElement('p');
                 noVotes.textContent = 'No valid answers found for this question.';
                 questionContainer.appendChild(noVotes);
            }
            consensusResults.appendChild(questionContainer);
        });
    }

    function showError(message) {
        errorBox.textContent = `Error: ${message}`;
        errorBox.classList.remove('hidden');
    }

    function hideError() {
        errorBox.classList.add('hidden');
    }

    function formatAnswer(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
    }
});

