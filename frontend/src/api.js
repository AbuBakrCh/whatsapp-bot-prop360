import axios from 'axios';

const BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export async function getConversations() {
  return axios.get(`${BASE}/conversations`).then(r => r.data);
}

export async function getChat(clientNumber) {
  return axios.get(`${BASE}/chats/${encodeURIComponent(clientNumber)}`).then(r => r.data);
}

export async function sendMessage(clientNumber, message, adminNumber) {
  return axios.post(`${BASE}/send`, { client_number: clientNumber, message, admin_number: adminNumber }).then(r => r.data);
}

export async function getClientConfig(clientNumber) {
  return axios.get(`${BASE}/get-client-config`, { params: { clientNumber } })
              .then(r => r.data);
}

export async function toggleClientBot(clientNumber, botEnabled) {
  return axios.post(`${BASE}/client-bot-toggle`, { clientNumber, botEnabled })
              .then(r => r.data);
}