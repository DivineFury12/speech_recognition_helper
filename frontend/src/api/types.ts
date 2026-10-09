import type { components, paths } from './schema'

export type Schemas = components['schemas']
export type AssistantTurn = Schemas['AssistantTurn']
export type Entry = Schemas['Entry']
export type EntryList = Schemas['EntryList']
export type TextMessageRequest = Schemas['TextMessageRequest']
export type ErrorResponse = Schemas['ErrorResponse']
export type ListEntriesQuery = paths['/v1/entries']['get']['parameters']['query']
export type HealthResponse = Schemas['HealthResponse']
