import { useEffect, useState } from 'react'
import { fetchImage } from '../api'

interface Props {
  src: string
  alt: string
  className?: string
}

/**
 * Loads an image that requires the bearer token, and revokes the object URL on
 * unmount so decrypted biometric images do not accumulate in browser memory.
 */
export function AuthedImage({ src, alt, className }: Props) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let created: string | null = null

    setObjectUrl(null)
    setError(null)

    fetchImage(src)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url)
          return
        }
        created = url
        setObjectUrl(url)
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message)
      })

    return () => {
      cancelled = true
      if (created) URL.revokeObjectURL(created)
    }
  }, [src])

  if (error) return <div className={`image-error ${className ?? ''}`}>{error}</div>
  if (!objectUrl) return <div className={`image-loading ${className ?? ''}`} aria-busy="true" />
  return <img src={objectUrl} alt={alt} className={className} />
}
