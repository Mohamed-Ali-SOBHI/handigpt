const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const speechButton = document.getElementById('speech-button');
const chatMessage = document.getElementById('chat-message');
const clearButton = document.getElementById('clear-button');
let messagehistory = [];   // Added message history array
const apiKey = 'sk-proj-F1l4Q9l7OZRo1WvZzkI2T3BlbkFJCuTXkY7K6KEpJbd0zFW5';

sendButton.addEventListener('click', async () => {
    const userMessage = userInput.value;
    userInput.value = '';
    addMessageToChat(' ', userMessage, 'user');
    const response = await generateResponse(userMessage);
    addMessageToChat(' ', response, 'model', userMessage);
});

clearButton.addEventListener('click', () => {
    chatMessage.innerHTML = '';
});

async function generateResponse(userMessage) {
    const openaiUrl = "https://api.openai.com/v1/chat/completions";  // OpenAI endpoint for chat completions
    const apiKey = ''

    messagehistory.push({role: 'user', content: userMessage});  // Add user message to history

    try {
        const response = await fetch(openaiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`  // Authorization header with the API key
            },
            body: JSON.stringify({
                model: "gpt-4o",  // Specify the model you want to use
                messages: messagehistory  // Provide the message history to the model
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        messagehistory.push({role: 'assistant', content: data.choices[0].message.content});  // Add model response to history
        console.log('Message history:', messagehistory);

        return data.choices[0].message.content;  // Accessing the message content from the response
    } catch (error) {
        console.error('Error fetching response from OpenAI:', error);
        return "Sorry, I couldn't fetch a response due to an error.";
    }
}

function addMessageToChat(sender, message, className, userMessage = '') {
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message', className);

    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.style.display = 'flex';
    messageElement.style.flexDirection = 'column'; // Changed to column for vertical stacking

    const textElement = document.createElement('span');
    textElement.classList.add('text');
    textElement.innerHTML = marked.parse(message);
    messageElement.appendChild(textElement);

    if (className === 'model') {
        const iconsContainer = document.createElement('div');
        iconsContainer.classList.add('icons-container'); // Added class for styling
        iconsContainer.style.display = 'flex';
        iconsContainer.style.gap = '10px'; // Space between icons
        iconsContainer.style.marginTop = '10px'; // Space between text and icons

        const speechElement = createIconButton('fas fa-volume-up', () => {
            speak(message);
        });
        iconsContainer.appendChild(speechElement);

        const copyElement = createIconButton('fas fa-copy', () => {
            navigator.clipboard.writeText(message);
        });
        iconsContainer.appendChild(copyElement);

        const redoElement = createIconButton('fas fa-redo', () => {
            userInput.value = userMessage;
            sendButton.click();
        });
        iconsContainer.appendChild(redoElement);

        messageElement.appendChild(iconsContainer); // Add icons below the message text
    }

    messageContainer.appendChild(messageElement); // Append message to container
    chatMessage.appendChild(messageContainer); // Append container to chat
    chatMessage.scrollTop = chatMessage.scrollHeight;
}

function createIconButton(iconClass, clickHandler) {
    const button = document.createElement('button');
    button.classList.add('icon-button');
    
    const icon = document.createElement('i');
    icon.className = iconClass;
    
    button.appendChild(icon);
    button.addEventListener('click', clickHandler);
    
    return button;
}

let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let silenceDetectionTimer;

speechButton.innerHTML = '<i class="fas fa-microphone"></i>';

speechButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording();
    } else {
        startRecording();
    }
});

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            audioContext = new AudioContext();
            analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            
            analyser.fftSize = 2048;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            mediaRecorder.start();
            speechButton.innerHTML = '<i class="fas fa-stop"></i>';

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(audioChunks);
                audioChunks = [];
                transcribeAudio(audioBlob);
            });

            // Détecter le silence
            function detectSilence() {
                analyser.getByteFrequencyData(dataArray);
                const sum = dataArray.reduce((a, b) => a + b, 0);
                const average = sum / bufferLength;
                
                if (average < 10) { // Ajustez ce seuil selon vos besoins
                    silenceDetectionTimer = setTimeout(() => {
                        stopRecording();
                    }, 1500); // Arrête après 1.5 secondes de silence
                } else {
                    clearTimeout(silenceDetectionTimer);
                }
                
                if (mediaRecorder.state === "recording") {
                    requestAnimationFrame(detectSilence);
                }
            }

            detectSilence();
        });
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
        speechButton.innerHTML = '<i class="fas fa-microphone"></i>';
        clearTimeout(silenceDetectionTimer);
    }
}

async function transcribeAudio(audioBlob) {
    speechButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

    const formData = new FormData();
    formData.append("file", audioBlob, "audio.webm");
    formData.append("model", "whisper-1");

    try {
        const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${apiKey}`
            },
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        userInput.value = data.text;
    } catch (error) {
        console.error("Error transcribing audio:", error);
        userInput.value = "Erreur lors de la transcription.";
    } finally {
        speechButton.innerHTML = '<i class="fas fa-microphone"></i>';
    }
}

async function speak(text) {
    try {
        const response = await fetch("https://api.openai.com/v1/audio/speech", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${apiKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model: "tts-1",
                voice: "alloy",
                input: text,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Créer un nouveau ReadableStream à partir de la réponse
        const reader = response.body.getReader();
        const stream = new ReadableStream({
            start(controller) {
                return pump();
                function pump() {
                    return reader.read().then(({ done, value }) => {
                        // Quand il n'y a plus de données à lire, fermer le stream
                        if (done) {
                            controller.close();
                            return;
                        }
                        // Sinon, envoyer le chunk au stream
                        controller.enqueue(value);
                        return pump();
                    });
                }
            }
        });

        // Créer un nouveau MediaSource
        const mediaSource = new MediaSource();
        const audio = new Audio();
        audio.src = URL.createObjectURL(mediaSource);

        mediaSource.addEventListener('sourceopen', async () => {
            const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
            const reader = stream.getReader();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                // Attendez que le sourceBuffer soit prêt avant d'ajouter des données
                if (!sourceBuffer.updating) {
                    sourceBuffer.appendBuffer(value);
                } else {
                    await new Promise(resolve => {
                        sourceBuffer.addEventListener('updateend', resolve, { once: true });
                    });
                    sourceBuffer.appendBuffer(value);
                }
            }

            mediaSource.endOfStream();
        });

        audio.play();
    } catch (error) {
        console.error('Erreur lors de la génération de la parole :', error);
    }
}
