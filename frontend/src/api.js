import axios from 'axios';

const BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export async function getConversations(page = 1, limit = 20) {
  const res = await fetch(`${BASE}/conversations?page=${page}&limit=${limit}`)
  if (!res.ok) throw new Error('Failed to fetch conversations')
  return res.json()
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
    const response = await axios.get(`${BASE}/utilities/duplicates-v2`);
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

export async function deleteProperties(payload) {
  try {
    const response = await axios.post(`${BASE}/properties/delete`, payload);
    return response.data;
  } catch (err) {
    console.error("Error deleting properties:", err);
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

export async function deleteContacts(payload) {
  try {
    const response = await axios.post(`${BASE}/contacts/delete`, payload);
    return response.data;
  } catch (err) {
    console.error("Error deleting contacts:", err);
    throw err;
  }
}


export async function generateActivitySummaries(payload) {
  const response = await axios.post(`${BASE}/forms/group-by-client-property`, payload);
  return response.data;
}

export async function getActivitySummaryProgress(jobId) {
  const response = await axios.get(`${BASE}/forms/activity-summary-progress/${jobId}`);
  return response.data;
}

export async function getPropertyActivitySummaries() {
  try {
    const response = await axios.get(`${BASE}/property-activity-summaries`);
    return response.data;
  } catch (err) {
    console.error("Error fetching property activity summaries:", err);
    throw err;
  }
}

export async function updateActivitySummaryStatus(summaryId, status) {
  try {
    const response = await axios.patch(`${BASE}/activity-summary/${summaryId}/status`, { status });
    return response.data;
  } catch (err) {
    console.error("Failed to update summary status:", err);
    return { error: err.message || "Failed to update status" };
  }
}

export async function addTimetables(payload) {
  try {
    const response = await axios.post(`${BASE}/timetables/add`, payload);
    return response.data;
  } catch (err) {
    console.error("Error adding timetables:", err);
    throw err;
  }
}

export async function deleteTimetables(payload) {
  try {
    const response = await axios.post(`${BASE}/timetables/delete`, payload);
    return response.data;
  } catch (err) {
    console.error("Error deleting timetables:", err);
    throw err;
  }
}

export async function addCashflows(payload) {
  try {
    const response = await axios.post(`${BASE}/cashflows/add`, payload);
    return response.data;
  } catch (err) {
    console.error("Error adding cashflows:", err);
    throw err;
  }
}

export async function deleteCashflows(payload) {
  try {
    const response = await axios.post(`${BASE}/cashflows/delete`, payload);
    return response.data;
  } catch (err) {
    console.error("Error deleting cashflows:", err);
    throw err;
  }
}


export async function getCashflowLedger(
  { groupType = "property", groupId, activityPage = 1, activityPageSize = 10 } = {}
) {
  try {
    const params = { activityPage, activityPageSize };
    if (groupType === "contact") {
      params.contactId = groupId;
    } else {
      params.propertyId = groupId;
    }
    const response = await axios.get(`${BASE}/cashflows/ledger`, { params });
    return response.data;
  } catch (err) {
    console.error("Error fetching ledger report:", err);
    throw err;
  }
}

export async function exportCashflowLedger({ groupType = "property", groupId } = {}) {
  try {
    const params = {};
    if (groupType === "contact") {
      params.contactId = groupId;
    } else {
      params.propertyId = groupId;
    }
    const response = await axios.get(`${BASE}/cashflows/ledger/export`, {
      params,
      responseType: "blob",
    });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `ledger-report-${groupType}-${groupId}.xlsx`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Error exporting cashflow ledger:", err);
    throw err;
  }
}

export async function mergeContacts(payload) {
  try {
    const response = await axios.post(`${BASE}/contacts/merge`, payload);
    return response.data;
  } catch (err) {
    console.error("Error merging contacts:", err);
    throw err;
  }
}

export async function controlJob(jobId, action) {
  try {
    const res = await axios.post(`${BASE}/jobs/${jobId}`, null, {
      params: { action },
    });
    return res.data;
  } catch (err) {
    console.error(`Failed to ${action} job ${jobId}:`, err);
    return { error: err.response?.data?.error || "Failed to control job" };
  }
}

export async function getExpiryJobControls() {
  try {
    const res = await axios.get(`${BASE}/job-control/expiry`);
    return res.data;
  } catch (err) {
    console.error("Error fetching expiry job controls:", err);
    throw err;
  }
}

export async function upsertExpiryJobRecipients(jobId, emails) {
  try {
    const res = await axios.post(`${BASE}/job-control/expiry/email-recipients`, {
      job_id: jobId,
      emails,
    });
    return res.data;
  } catch (err) {
    console.error(`Error updating recipients for job ${jobId}:`, err);
    throw err;
  }
}

export async function startSpitogatosCrawler(payload) {
  try {
    const response = await axios.post(`${BASE}/crawler/spitogatos`, payload);
    return response.data;
  } catch (err) {
    return { error: err.response?.data?.detail || "Failed to start crawler" };
  }
}

export async function stopSpitogatosCrawler() {
  try {
    const response = await axios.post(`${BASE}/crawler/spitogatos/stop`);
    return response.data;
  } catch (err) {
    return { error: err.response?.data?.detail || "Failed to stop crawler" };
  }
}

export async function addPropertyFilter(payload) {
  try {
    const response = await axios.post(
      `${BASE}/property-filters/add`,
      payload
    );
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to save filters",
    };
  }
}

export async function getPropertyFilter(email) {
  try {
    const response = await axios.get(
      `${BASE}/property-filters/${email}`
    );
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to fetch filter",
    };
  }
}

export async function getSharePropertyJobStatus() {
  try {
    const res = await axios.get(`${BASE}/jobs/share-property`);
    return res.data;
  } catch (err) {
    return { error: "Failed to fetch job status." };
  }
}

export async function updateSharePropertyJobStatus(status) {
  try {
    const res = await axios.patch(`${BASE}/jobs/share-property`, {
      status, // "enable" or "disable"
    });
    return res.data;
  } catch (err) {
    return { error: "Request failed." };
  }
}

export async function getAllPropertyFilters(page = 1, limit = 10) {
  try {
    const response = await axios.get(
      `${BASE}/property-filters`,
      {
        params: { page, page_size: limit }
      }
    );

    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to fetch filters",
    };
  }
}

export async function deletePropertyFilter(email) {
  try {
    const res = await axios.delete(`${BASE}/property-filters/${email}`);
    return res.data;
  } catch (err) {
    return { error: "Failed to delete filter" };
  }
}

export async function searchGroupUsers(q, limit = 10, excludeIds = []) {
  try {
    const response = await axios.get(`${BASE}/groups/users/search`, {
      params: {
        q,
        limit,
        exclude: excludeIds.length ? excludeIds.join(",") : undefined,
      },
    });
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to search users",
    };
  }
}

export async function getGroups(page = 1, limit = 10) {
  try {
    const response = await axios.get(`${BASE}/groups`, {
      params: { page, page_size: limit },
    });
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to fetch groups",
    };
  }
}

export async function getGroup(groupId) {
  try {
    const response = await axios.get(
      `${BASE}/groups/${encodeURIComponent(groupId)}`
    );
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to fetch group",
    };
  }
}

export async function createGroup(payload) {
  try {
    const response = await axios.post(`${BASE}/groups`, payload);
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to create group",
    };
  }
}

export async function updateGroup(groupId, payload) {
  try {
    const response = await axios.put(
      `${BASE}/groups/${encodeURIComponent(groupId)}`,
      payload
    );
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to update group",
    };
  }
}

export async function deleteGroup(groupId) {
  try {
    const res = await axios.delete(
      `${BASE}/groups/${encodeURIComponent(groupId)}`
    );
    return res.data;
  } catch (err) {
    return { error: "Failed to delete group" };
  }
}

export async function searchGroups(q, limit = 500, excludeIds = []) {
  try {
    const response = await axios.get(`${BASE}/groups/search`, {
      params: {
        q,
        limit,
        exclude: excludeIds.length ? excludeIds.join(",") : undefined,
      },
    });
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to search groups",
    };
  }
}

export async function shareWithGroups(payload) {
  try {
    const response = await axios.post(`${BASE}/groups/share`, payload);
    return response.data;
  } catch (err) {
    return {
      error:
        err.response?.data?.error ||
        err.response?.data?.detail ||
        "Failed to share",
    };
  }
}