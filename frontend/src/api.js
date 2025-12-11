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

export const processBankStatementsFromDrive = async (folderLink, authToken, mapping) => {
  const payload = {
    folder_link: folderLink,
    auth_token: authToken,
    mapping
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

export async function getDuplicates() {
  try {
    const response = await axios.get(`${BASE}/utilities/duplicates`);
    return response.data;
  } catch (err) {
    console.error("Error fetching duplicates:", err);
    throw err;
  }
}

export async function generateClientMessages(date, prompt, merchantId) {
  try {
    const response = await axios.post(`${BASE}/utilities/activity/client-messages`, {
      date,
      prompt,
      merchantId
    });
    return response.data;
  } catch (err) {
    console.error("Error generating client messages:", err);
    throw err;
  }
}

export async function sendActivityEmails(merchantId, date) {
  try {
    const response = await axios.post(
      `${BASE}/utilities/activity/send-emails`,
      {
        merchantId,
        date
      }
    );
    return response.data;
  } catch (err) {
    console.error("Error sending emails:", err);
    throw err;
  }
}

export async function getPrompt(promptId) {
  try {
    const response = await axios.get(`${BASE}/utilities/prompts/get`, { params: { prompt_id: promptId } });
    return response.data;
  } catch (err) {
    console.error("Error fetching prompt:", err);
    throw err;
  }
}

export async function savePrompt(promptId, promptText) {
  try {
    const response = await axios.post(`${BASE}/utilities/prompts/save`, { prompt_id: promptId, prompt_text: promptText });
    return response.data;
  } catch (err) {
    console.error("Error saving prompt:", err);
    throw err;
  }
}

export async function addProperties(payload) {
  try {
    const response = await axios.post(`${BASE}/properties/add`, payload);
    return response.data;
  } catch (err) {
    console.error("Error adding properties:", err);
    throw err;
  }
}

export async function addContacts(payload) {
  try {
    const response = await axios.post(`${BASE}/contacts/add`, payload);
    return response.data;
  } catch (err) {
    console.error("Error adding contacts:", err);
    throw err;
  }
}