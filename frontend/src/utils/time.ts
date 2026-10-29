export const timezone: string = Intl.DateTimeFormat().resolvedOptions().timeZone

export function isoDate(d: Date, tz = timezone): string {
  return new Intl.DateTimeFormat('en-CA', { timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit' }).format(d)
}

export function addDays(d: Date, n: number): Date {
  const r = new Date(d)
  r.setDate(r.getDate() + n)
  return r
}

export function formatDate(date: string): string {
  const [y, m, d] = date.split('-').map(Number)
  const options: Intl.DateTimeFormatOptions = { weekday: 'short', day: 'numeric', month: 'short' }
  if (y !== Number(isoDate(new Date()).slice(0, 4))) options.year = 'numeric'
  return new Date(y, m - 1, d).toLocaleDateString('ru-RU', options)
}
