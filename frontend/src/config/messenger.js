export function isMessengerEnabled() {
  return import.meta.env.VITE_WHATSAPP_MESSENGER_ENABLED === 'true'
}
