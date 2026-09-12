// Thin re-export so every icon in the app shares one stroke weight and size
// by default, rather than each call site guessing at props.
import {
  ArchiveIcon,
  ArrowLeftIcon,
  ArrowRightIcon,
  CheckIcon,
  CircleIcon,
  CompassIcon,
  LoaderCircleIcon,
  StampIcon,
  TriangleAlertIcon,
  XIcon,
} from 'lucide-react';

const DEFAULTS = { strokeWidth: 1.5, size: 18 };

function icon(Component) {
  return (props) => <Component {...DEFAULTS} {...props} />;
}

export const Icon = {
  Archive: icon(ArchiveIcon),
  ArrowLeft: icon(ArrowLeftIcon),
  ArrowRight: icon(ArrowRightIcon),
  Check: icon(CheckIcon),
  Circle: icon(CircleIcon),
  Compass: icon(CompassIcon),
  Loading: icon(LoaderCircleIcon),
  Stamp: icon(StampIcon),
  Warning: icon(TriangleAlertIcon),
  Close: icon(XIcon),
};
