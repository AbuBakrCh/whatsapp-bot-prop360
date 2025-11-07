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

export async function getDetails(client) {
  return axios
    .get(`${BASE}/details`, { params: { client } })
    .then((r) => r.data);
}

export async function updateDetails(client, name, info) {
  return axios
    .put(`${BASE}/details`, { client, name, info })
    .then((r) => r.data);
}

export async function sendBulkEmailFile(driveLink) {
  const formData = new FormData();
  formData.append("drive_link", driveLink);

  const response = await axios.post(`${BASE}/send-bulk-email`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
}

export const processBankStatementsFromDrive = async (folderLink, authToken) => {
  const payload = {
    folder_link: folderLink,
    auth_token: authToken,
  };

  const response = await axios.post(
    `${BASE}/bank-statements/from-drive-folder`,
    payload,
    {
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    }
  );

  return response.data;
};