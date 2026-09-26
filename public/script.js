const thoughts = document.querySelector('#thoughts');
const submit = document.querySelector('#submit');
const counter = document.querySelector('#counter');
const result = document.querySelector('#result');
const error = document.querySelector('#error');
const loading = document.querySelector('#loading');

function renderMusic(music) {
	document.querySelector('#music-list').innerHTML = music.map(item => `<a class="resource-link music-item" href="${item.spotify_url}" target="_blank" rel="noreferrer"><span class="resource-icon">&#9835;</span><div><p class="resource-label">Your soundtrack · ${item.mood}</p><p>${item.title} <span class="music-description">${item.artist}</span></p><small>${item.reason}</small><strong>Listen on Spotify &#8599;</strong></div></a>`).join('');
}

thoughts.addEventListener('input', () => {
	counter.textContent = `${thoughts.value.length.toLocaleString()} / 2,000`;
});

submit.addEventListener('click', async () => {
	error.hidden = true;
	if (!thoughts.value.trim()) {
		error.textContent = 'Write a few words first, whenever you are ready.';
		error.hidden = false;
		thoughts.focus();
		return;
	}
	submit.disabled = true;
	loading.hidden = false;
	result.hidden = true;
	try {
		const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: thoughts.value }) });
		const data = await response.json();
		if (!response.ok) throw new Error(data.error || 'Unable to process that message.');
		document.querySelector('#sentiment span:last-child').textContent = data.sentiment;
		document.querySelector('#sentiment').className = `sentiment ${data.sentiment}`;
		document.querySelector('#acknowledgement').textContent = data.acknowledgement;
		document.querySelector('#message').textContent = data.message;
		document.querySelector('#suggestion').textContent = data.suggestion;
		document.querySelector('#follow-up').textContent = data.follow_up;
		document.querySelector('#quote').textContent = data.quote;
		document.querySelector('#exercise-title').textContent = data.exercise.title;
		document.querySelector('#exercise-instructions').textContent = data.exercise.instructions;
		renderMusic(data.music);
		document.querySelector('#breathing-card').hidden = !data.breathing;
		const safety = document.querySelector('#safety');
		safety.hidden = !data.safety;
		safety.textContent = data.safety ? 'For immediate danger, contact local emergency services. A trusted person or local crisis support can also help you connect with human care.' : '';
		result.hidden = false;
		result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
	} catch (requestError) {
		error.textContent = requestError.message;
		error.hidden = false;
	} finally {
		submit.disabled = false;
		loading.hidden = true;
	}
});

document.querySelector('#reset-chat').addEventListener('click', async () => {
	await fetch('/reset', { method: 'POST' });
	thoughts.value = '';
	counter.textContent = '0 / 2,000';
	error.hidden = true;
	result.hidden = true;
	thoughts.focus();
});
